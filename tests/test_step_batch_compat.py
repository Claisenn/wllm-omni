"""Tests for denoise-step batch compatibility.

These cover the three layers that jointly decided whether two diffusion
requests could ever share a forward batch: the scheduler's admission check,
the executor's grouping key, and ModelRunner's grouping.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from wllm_omni.engine.model_runner import ModelRunner
from wllm_omni.model_types import ModelParadigm
from wllm_omni.request import OmniRequest
from wllm_omni.sampling_params import PRESETS, clone_sampling_params
from wllm_omni.sched.interface import RequestStatus, StepBatchSamplingParamsKey
from wllm_omni.sched.step_scheduler import StepScheduler


def make_request(**overrides) -> OmniRequest:
    sampling = clone_sampling_params(PRESETS["quality"])
    for name, value in overrides.items():
        setattr(sampling, name, value)
    return OmniRequest(prompt="a cat", image="unused.png", sampling_params=sampling)


class TestStepBatchSamplingParamsKey:
    def test_identical_params_compare_equal(self):
        assert StepBatchSamplingParamsKey.from_sampling_params(
            make_request().sampling_params
        ) == StepBatchSamplingParamsKey.from_sampling_params(make_request().sampling_params)

    @pytest.mark.parametrize(
        "field, value",
        [
            ("height", 512),
            ("width", 512),
            ("num_frames", 33),
            ("fps", 24),
            ("guidance_scale", 7.5),
            ("flow_shift", 5.0),
            ("num_inference_steps", 20),
        ],
    )
    def test_shape_and_operator_fields_split_the_batch(self, field, value):
        base = StepBatchSamplingParamsKey.from_sampling_params(make_request().sampling_params)
        other = StepBatchSamplingParamsKey.from_sampling_params(make_request(**{field: value}).sampling_params)
        assert base != other, f"{field} must be part of the batch-compatibility key"

    @pytest.mark.parametrize(
        "field, value",
        [
            # Only picks the initial noise, which is sampled per request and stacked.
            ("seed", 1234),
            # Only changes the contents of a fixed-length embedding, not its shape.
            ("negative_prompt", "something else entirely"),
        ],
    )
    def test_request_local_fields_do_not_split_the_batch(self, field, value):
        base = StepBatchSamplingParamsKey.from_sampling_params(make_request().sampling_params)
        other = StepBatchSamplingParamsKey.from_sampling_params(make_request(**{field: value}).sampling_params)
        assert base == other, f"{field} is request-local and must not split a batch"

    def test_key_is_hashable_so_it_can_index_a_group(self):
        key = StepBatchSamplingParamsKey.from_sampling_params(make_request().sampling_params)
        assert {key: "group"}[replace(key)] == "group"


class TestSchedulerAdmission:
    def test_compatible_requests_are_scheduled_together(self):
        scheduler = StepScheduler(max_num_running_reqs=2)
        scheduler.add_request(make_request())
        scheduler.add_request(make_request())

        out = scheduler.schedule()

        assert out.num_scheduled_reqs == 2
        assert out.num_waiting_reqs == 0

    def test_incompatible_request_stays_queued(self):
        scheduler = StepScheduler(max_num_running_reqs=4)
        first = make_request()
        scheduler.add_request(first)
        scheduler.add_request(make_request(height=512))

        out = scheduler.schedule()

        assert out.scheduled_req_ids == [first.request_id]
        assert out.num_waiting_reqs == 1

    def test_queued_request_runs_once_the_batch_drains(self):
        scheduler = StepScheduler(max_num_running_reqs=4)
        scheduler.add_request(make_request())
        queued = make_request(height=512)
        scheduler.add_request(queued)

        first_out = scheduler.schedule()
        scheduler.finish_requests(first_out.scheduled_req_ids, RequestStatus.FINISHED_COMPLETED)
        second_out = scheduler.schedule()

        assert second_out.scheduled_req_ids == [queued.request_id]

    def test_head_of_line_blocking_is_intentional(self):
        """A compatible request behind an incompatible one also waits.

        The waiting queue is FIFO and admission stops at the first request that
        cannot join, rather than scanning past it. This keeps arrival order
        meaningful; reordering would need an explicit fairness policy.
        """
        scheduler = StepScheduler(max_num_running_reqs=4)
        head = make_request()
        scheduler.add_request(head)
        scheduler.add_request(make_request(height=512))
        scheduler.add_request(make_request())

        out = scheduler.schedule()

        assert out.scheduled_req_ids == [head.request_id]
        assert out.num_waiting_reqs == 2


class FakeWanPipeline:
    """Minimal stand-in satisfying the step-execution contract."""

    def __init__(self):
        self.config = type("Config", (), {"enable_profiling": False, "use_cpu_offload": False})()

    def prepare_encode(self, state):
        return state

    def denoise_step(self, state):
        return None

    def step_scheduler(self, state, noise_pred):
        return None

    def post_decode(self, state):
        return None


class TestExecutorGrouping:
    @staticmethod
    def _executor():
        from wllm_omni.models.diffusion_executor import DiffusionExecutor

        return DiffusionExecutor(FakeWanPipeline())

    def test_batch_key_ignores_request_identity(self):
        executor = self._executor()
        left = executor.init_state("sched-a", make_request())
        right = executor.init_state("sched-b", make_request())

        assert executor.batch_key(left) == executor.batch_key(right)

    def test_batch_key_ignores_step_index(self):
        executor = self._executor()
        state = executor.init_state("sched-a", make_request())
        before = executor.batch_key(state)
        state.payload.step_index = 7

        assert executor.batch_key(state) == before

    def test_batch_key_still_separates_incompatible_requests(self):
        executor = self._executor()
        left = executor.init_state("sched-a", make_request())
        right = executor.init_state("sched-b", make_request(height=512))

        assert executor.batch_key(left) != executor.batch_key(right)

    def test_model_runner_groups_compatible_states_together(self):
        executor = self._executor()
        runner = ModelRunner(config=None, executors=[executor])
        states = [
            executor.init_state("sched-a", make_request()),
            executor.init_state("sched-b", make_request()),
            executor.init_state("sched-c", make_request(height=512)),
        ]

        groups = runner._group_states(states)

        assert sorted(len(group) for group in groups) == [1, 2]
        assert all(state.paradigm is ModelParadigm.DIFFUSION for group in groups for state in group)


class TestSchedulerSeamWithAR:
    """The AR and diffusion schedulers share BaseScheduler.

    Batch compatibility is a diffusion property. Putting it on the shared base
    class silently gated AR requests on diffusion sampling parameters, so these
    pin the boundary: StepScheduler constrains, RequestScheduler does not.
    """

    @staticmethod
    def _ar_request(**overrides):
        from wllm_omni.model_types import ModelParadigm

        request = make_request(**overrides)
        request.model_paradigm = ModelParadigm.AUTOREGRESSIVE
        return request

    def test_ar_scheduler_is_not_gated_on_diffusion_params(self):
        from wllm_omni.sched.request_scheduler import RequestScheduler

        scheduler = RequestScheduler(max_num_running_reqs=4)
        scheduler.add_request(self._ar_request())
        scheduler.add_request(self._ar_request(height=512))

        out = scheduler.schedule()

        assert out.num_scheduled_reqs == 2, "resolution is meaningless to the AR stage"
        assert out.num_waiting_reqs == 0

    def test_ar_scheduler_builds_no_batch_key(self):
        from wllm_omni.sched.request_scheduler import RequestScheduler

        scheduler = RequestScheduler(max_num_running_reqs=2)
        sched_req_id = scheduler.add_request(self._ar_request())

        assert scheduler.get_request_state(sched_req_id).sampling_params_key is None

    def test_step_scheduler_still_builds_a_batch_key(self):
        scheduler = StepScheduler(max_num_running_reqs=2)
        sched_req_id = scheduler.add_request(make_request())

        assert scheduler.get_request_state(sched_req_id).sampling_params_key is not None
