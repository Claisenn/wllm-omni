from __future__ import annotations

from collections import deque

from wllm_omni.request import OmniRequest
from wllm_omni.sched.interface import (
    RequestStatus,
    ScheduledRequest,
    SchedulerInterface,
    SchedulerOutput,
    SchedulerRequestState,
    StepBatchSamplingParamsKey,
)


class BaseScheduler(SchedulerInterface):

    def __init__(self, max_num_running_reqs: int = 1):
        self._request_states: dict[str, SchedulerRequestState] = {}
        self._request_id_to_sched_req_id: dict[str, str] = {}
        self._step_id = 0
        self._waiting: deque[str] = deque()
        self._running: list[str] = []
        self._finished_req_ids: set[str] = set()
        self._running_sampling_params_key: StepBatchSamplingParamsKey | None = None
        self.max_num_running_reqs = max_num_running_reqs

    def add_request(self, request: OmniRequest) -> str:
        sched_req_id = self._make_sched_req_id(request)
        state = self._make_request_state(sched_req_id, request)
        self._request_states[sched_req_id] = state
        self._request_id_to_sched_req_id[request.request_id] = sched_req_id
        self._waiting.append(sched_req_id)
        return sched_req_id

    def _make_request_state(self, sched_req_id: str, request: OmniRequest) -> SchedulerRequestState:
        return SchedulerRequestState(
            sched_req_id=sched_req_id,
            req=request,
            sampling_params_key=self._build_sampling_params_key(request),
        )

    def _build_sampling_params_key(self, request: OmniRequest) -> StepBatchSamplingParamsKey | None:
        """Batch-compatibility key for this scheduler, or None for no constraint.

        The base scheduler imposes none: whether two requests may run together
        is a property of how a paradigm batches, not of queueing. Subclasses
        serving a paradigm with a homogeneous-batch requirement override this.
        """
        return None

    def _can_schedule_waiting(self, state: SchedulerRequestState) -> bool:
        """Admit a waiting request only if it can share a batch with the running set.

        Homogeneous batching: rather than padding heterogeneous requests to a
        common shape, the scheduler only groups requests that are already
        compatible and leaves the rest queued for a later batch. A scheduler
        that returns no key from _build_sampling_params_key opts out entirely.
        """
        if not self._running or state.sampling_params_key is None:
            return True
        current_key = self._current_sampling_params_key()
        return current_key is not None and current_key == state.sampling_params_key

    def _current_sampling_params_key(self) -> StepBatchSamplingParamsKey | None:
        if self._running_sampling_params_key is not None or not self._running:
            return self._running_sampling_params_key
        state = self._request_states.get(self._running[0])
        self._running_sampling_params_key = None if state is None else state.sampling_params_key
        return self._running_sampling_params_key

    def schedule(self) -> SchedulerOutput:
        scheduled_reqs: list[ScheduledRequest] = []

        for sched_req_id in self._running:
            state = self._request_states.get(sched_req_id)
            if state is not None:
                scheduled_reqs.append(ScheduledRequest.from_state(state, is_new=False))

        while self._waiting and len(self._running) < self.max_num_running_reqs:
            sched_req_id = self._waiting[0]
            state = self._request_states.get(sched_req_id)
            if state is None:
                self._waiting.popleft()
                continue
            if not self._can_schedule_waiting(state):
                break
            self._waiting.popleft()
            was_new = state.status == RequestStatus.WAITING
            if not self._running:
                self._running_sampling_params_key = state.sampling_params_key
            state.status = RequestStatus.RUNNING
            self._running.append(sched_req_id)
            scheduled_reqs.append(ScheduledRequest.from_state(state, is_new=was_new))

        out = SchedulerOutput(
            step_id=self._step_id,
            scheduled_reqs=scheduled_reqs,
            finished_req_ids=set(self._finished_req_ids),
            num_running_reqs=len(self._running),
            num_waiting_reqs=len(self._waiting),
        )
        self._step_id += 1
        self._finished_req_ids.clear()
        return out

    def has_requests(self) -> bool:
        return bool(self._waiting or self._running)

    def get_request_state(self, sched_req_id: str) -> SchedulerRequestState | None:
        return self._request_states.get(sched_req_id)

    def get_sched_req_id(self, request_id: str) -> str | None:
        return self._request_id_to_sched_req_id.get(request_id)

    def pop_request_state(self, sched_req_id: str) -> SchedulerRequestState | None:
        state = self._request_states.pop(sched_req_id, None)
        if state is not None and self._request_id_to_sched_req_id.get(state.req.request_id) == sched_req_id:
            self._request_id_to_sched_req_id.pop(state.req.request_id, None)
        return state

    def preempt_request(self, sched_req_id: str) -> bool:
        if sched_req_id not in self._request_states:
            return False
        if sched_req_id in self._running:
            self._running.remove(sched_req_id)
            self._waiting.appendleft(sched_req_id)
            self._request_states[sched_req_id].status = RequestStatus.PREEMPTED
            self._reset_key_if_idle()
            return True
        return False

    def _reset_key_if_idle(self) -> None:
        """Release the batch key once no request is running, so the next
        scheduling round is free to start a batch with different parameters."""
        if not self._running:
            self._running_sampling_params_key = None

    def finish_requests(self, sched_req_ids: str | list[str], status: RequestStatus) -> None:
        assert RequestStatus.is_finished(status)
        if isinstance(sched_req_ids, str):
            sched_req_ids = [sched_req_ids]
        statuses = {sched_req_id: status for sched_req_id in sched_req_ids}
        self._finish_requests(statuses)

    def close(self) -> None:
        self._request_states.clear()
        self._request_id_to_sched_req_id.clear()
        self._waiting.clear()
        self._running.clear()
        self._finished_req_ids.clear()
        self._running_sampling_params_key = None

    def _finish_requests(
        self,
        statuses: dict[str, RequestStatus],
        errors: dict[str, str | None] | None = None,
    ) -> set[str]:
        if not statuses:
            return set()
        finished_req_ids: set[str] = set()
        running_to_remove: set[str] = set()
        waiting_to_remove: set[str] = set()
        for sched_req_id, status in statuses.items():
            state = self._request_states.get(sched_req_id)
            if state is None or state.is_finished():
                continue
            finished_req_ids.add(sched_req_id)
            if sched_req_id in self._running:
                running_to_remove.add(sched_req_id)
            if sched_req_id in self._waiting:
                waiting_to_remove.add(sched_req_id)
        if running_to_remove:
            self._running = [req_id for req_id in self._running if req_id not in running_to_remove]
            self._reset_key_if_idle()
        if waiting_to_remove:
            self._waiting = deque(req_id for req_id in self._waiting if req_id not in waiting_to_remove)
        for sched_req_id in finished_req_ids:
            state = self._request_states[sched_req_id]
            state.status = statuses[sched_req_id]
            state.error = None if errors is None else errors.get(sched_req_id)
        self._finished_req_ids |= finished_req_ids
        return finished_req_ids

    def _finalize_update_from_output(
        self,
        sched_output: SchedulerOutput,
        statuses: dict[str, RequestStatus],
        errors: dict[str, str | None] | None = None,
    ) -> set[str]:
        finished_req_ids = {
            sched_req_id for sched_req_id in sched_output.scheduled_req_ids if sched_req_id in self._finished_req_ids
        }
        finished_req_ids |= self._finish_requests(statuses, errors)
        return finished_req_ids
