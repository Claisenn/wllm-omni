from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Sequence

from wllm_omni.model_types import ModelParadigm
from wllm_omni.worker.utils import (
    ExecutorCapability,
    ForwardBatch,
    ModelForwardOutput,
    RequestState,
    RunnerOutput,
)

if TYPE_CHECKING:
    from wllm_omni.request import OmniRequest

STEP_EXECUTION_METHODS = ("prepare_encode", "denoise_step", "step_scheduler", "post_decode")


def supports_step_execution(pipeline: Any, methods: Sequence[str] = STEP_EXECUTION_METHODS) -> bool:
    """Duck-type check for whether a pipeline implements a step-execution contract.

    Defaults to the diffusion contract so existing call sites are unaffected;
    other paradigms (e.g. AR prefill/decode) pass their own ``methods`` tuple.
    """
    return all(callable(getattr(pipeline, name, None)) for name in methods)


class ModelExecutor(ABC):
    """Executor contract used by ModelRunner.

    The runner owns request lifecycle and batching orchestration. Concrete
    executors own model-family details such as KV cache, diffusion latents,
    multimodal feature caches, or world-model rollout state.
    """

    paradigm: ModelParadigm
    capabilities: frozenset[ExecutorCapability] = frozenset()

    @abstractmethod
    def init_state(self, sched_req_id: str, request: OmniRequest) -> RequestState:
        pass

    @abstractmethod
    def batch_key(self, state: RequestState) -> tuple:
        pass

    @abstractmethod
    def build_forward_batch(self, states: list[RequestState]) -> ForwardBatch:
        pass

    @abstractmethod
    def forward(self, batch: ForwardBatch) -> ModelForwardOutput:
        pass

    def update_states(self, states: list[RequestState], output: ModelForwardOutput) -> None:
        """Write forward results back into the runner-owned request states.

        Progress must round-trip through ``ModelForwardOutput.payload`` (not
        rely on states aliasing the batch payload) so executors survive a
        worker/serialization boundary. Both current executors share this exact
        logic; override only if a model family needs different bookkeeping.
        """
        output_by_req_id = {item.req_id: item for item in output.outputs}
        for state in states:
            item = output_by_req_id.get(state.sched_req_id)
            if item is None:
                continue
            if output.payload is not None:
                state.payload = output.payload
                state.initialized = True
            if item.error is not None:
                state.error = item.error
                state.finished = True
            if item.step_index is not None:
                state.step_index = item.step_index
            if item.finished:
                state.finished = True

    def collect_outputs(
        self,
        states: list[RequestState],
        output: ModelForwardOutput,
    ) -> list[RunnerOutput]:
        """Default: forward() already attached results to its RunnerOutputs."""
        return output.outputs

    def release(self, state: RequestState) -> None:
        state.payload = None


class ExecutorRegistry:
    """Paradigm-indexed executor registry used by the generic runner."""

    def __init__(self, executors: list[ModelExecutor]):
        if not executors:
            raise ValueError("ExecutorRegistry requires at least one executor.")

        self._executors: dict[ModelParadigm, ModelExecutor] = {}
        for executor in executors:
            if executor.paradigm in self._executors:
                raise ValueError(f"Duplicate executor registered for paradigm={executor.paradigm}.")
            self._executors[executor.paradigm] = executor
        self.default_executor = executors[0]

    @property
    def executors(self) -> dict[ModelParadigm, ModelExecutor]:
        return dict(self._executors)

    def resolve_request(self, request: OmniRequest) -> ModelExecutor:
        paradigm = getattr(request, "model_paradigm", None)
        if paradigm is None:
            return self.default_executor
        return self.resolve_paradigm(paradigm)

    def resolve_state(self, state: RequestState) -> ModelExecutor:
        return self.resolve_paradigm(state.paradigm)

    def resolve_paradigm(self, paradigm: ModelParadigm | str) -> ModelExecutor:
        if isinstance(paradigm, str):
            try:
                paradigm = ModelParadigm(paradigm)
            except ValueError as exc:
                known = ", ".join(item.value for item in self._executors)
                raise ValueError(f"Unknown model paradigm={paradigm!r}; registered paradigms: {known}.") from exc
        executor = self._executors.get(paradigm)
        if executor is None:
            known = ", ".join(item.value for item in self._executors)
            raise ValueError(f"No executor registered for paradigm={paradigm.value}; registered paradigms: {known}.")
        return executor
