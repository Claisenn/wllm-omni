from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from wllm_omni.model_types import ModelParadigm
from wllm_omni.models import ModelExecutor, supports_step_execution
from wllm_omni.models.ar_pipeline import ARPipeline, ARTextOutput, IdentityARPipeline
from wllm_omni.worker.utils import (
    ExecutionPhase,
    ExecutorCapability,
    ForwardBatch,
    ModelForwardOutput,
    RequestState,
    RunnerOutput,
)

if TYPE_CHECKING:
    from wllm_omni.request import OmniRequest

# The full stepwise contract, including finalize: the executor calls all
# three, so a pipeline offering only a subset must fall back to generate().
AR_STEP_EXECUTION_METHODS = ("prefill", "decode_step", "finalize")


@dataclass(slots=True)
class ARState:
    """Executor-side payload: the request plus opaque pipeline decode state.

    ``decode`` is whatever the pipeline's ``prefill``/``decode_step`` return;
    the executor only reads its ``finished`` attribute. It holds the KV cache,
    so it is dropped as soon as ``finalize`` has produced ``output``.
    """

    request: OmniRequest
    decode: Any = None
    output: ARTextOutput | None = None
    step_index: int = 0


class ARExecutor(ModelExecutor):
    """AR executor for the mini-Omni runtime.

    When the pipeline implements the full stepwise contract
    (``AR_STEP_EXECUTION_METHODS``, e.g. TransformersARPipeline), each
    ``forward()`` call advances exactly one step -- PREPARE runs prefill, STEP
    runs one KV-cache decode step -- mirroring DiffusionExecutor's
    PREPARE-then-STEP rhythm, except termination is decided by the pipeline
    (EOS / token budget) instead of a fixed step count. The decode state
    (including the KV cache) travels in ``ModelForwardOutput.payload`` and is
    written back in ``update_states``, so progress does not rely on in-process
    aliasing between the runner's state and the forward batch.

    Pipelines that only implement ``generate`` (e.g. IdentityARPipeline) run
    to completion in a single FINALIZE-phase forward call, unchanged.
    """

    paradigm = ModelParadigm.AUTOREGRESSIVE
    # Note: nothing reads .capabilities yet anywhere in the runtime; this set
    # is kept as-is from V0 rather than extended with more ornamental flags.
    capabilities = frozenset({ExecutorCapability.STEPWISE, ExecutorCapability.STREAMING})

    def __init__(self, pipeline: ARPipeline | None = None):
        self.pipeline = pipeline or IdentityARPipeline()
        self._stepwise = supports_step_execution(self.pipeline, AR_STEP_EXECUTION_METHODS)

    def init_state(self, sched_req_id: str, request: OmniRequest) -> RequestState:
        return RequestState(
            req_id=request.request_id,
            sched_req_id=sched_req_id,
            paradigm=self.paradigm,
            payload=ARState(request=request),
        )

    def batch_key(self, state: RequestState) -> tuple:
        # One request per forward batch: AR decode is per-sequence anyway
        # (the pipeline API takes a single request), and per-request batches
        # keep the batch phase well-defined -- same shape as DiffusionExecutor.
        return (self.paradigm.value, state.sched_req_id)

    def build_forward_batch(self, states: list[RequestState]) -> ForwardBatch:
        if len(states) != 1:
            raise ValueError(f"ARExecutor supports exactly one request per forward batch, got {len(states)}.")
        state = states[0]
        if not self._stepwise:
            # Monolithic pipelines produce the final output in one call.
            phase = ExecutionPhase.FINALIZE
        elif not state.initialized:
            phase = ExecutionPhase.PREPARE
        else:
            phase = ExecutionPhase.STEP
        return ForwardBatch(
            paradigm=self.paradigm,
            req_ids=[state.sched_req_id],
            phase=phase,
            payload=self._state_payload(state),
        )

    def forward(self, batch: ForwardBatch) -> ModelForwardOutput:
        if batch.paradigm != self.paradigm:
            raise ValueError(f"ARExecutor cannot run batch for paradigm={batch.paradigm}.")
        if len(batch.req_ids) != 1:
            raise ValueError(f"ARExecutor supports exactly one request per forward batch, got {len(batch.req_ids)}.")

        ar_state = self._batch_payload(batch)
        ar_state.step_index += 1

        if batch.phase == ExecutionPhase.FINALIZE:
            ar_state.output = self.pipeline.generate(ar_state.request)
        elif batch.phase == ExecutionPhase.PREPARE:
            ar_state.decode = self.pipeline.prefill(ar_state.request)
        else:
            if ar_state.decode is None:
                raise ValueError("ARExecutor got a STEP batch without prefilled decode state.")
            ar_state.decode = self.pipeline.decode_step(ar_state.decode)

        if ar_state.output is None and ar_state.decode is not None and ar_state.decode.finished:
            ar_state.output = self.pipeline.finalize(ar_state.decode)
            # The KV cache is dead weight once the output exists; drop it now
            # instead of keeping it alive until release().
            ar_state.decode = None

        finished = ar_state.output is not None
        runner_output = RunnerOutput(
            req_id=batch.req_ids[0],
            step_index=ar_state.step_index,
            finished=finished,
            result=ar_state.output if finished else None,
        )
        return ModelForwardOutput(outputs=[runner_output], payload=ar_state)

    def update_states(self, states: list[RequestState], output: ModelForwardOutput) -> None:
        output_by_req_id = {item.req_id: item for item in output.outputs}
        for state in states:
            item = output_by_req_id.get(state.sched_req_id)
            if item is None:
                continue
            if output.payload is not None:
                # Progress must round-trip through the forward output, not
                # rely on state.payload aliasing the batch payload -- same
                # contract DiffusionExecutor honors for worker boundaries.
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
        # forward() already attached result to the finishing RunnerOutput.
        return output.outputs

    def release(self, state: RequestState) -> None:
        state.payload = None

    @staticmethod
    def _state_payload(state: RequestState) -> ARState:
        if not isinstance(state.payload, ARState):
            raise TypeError(f"Expected ARState payload, got {type(state.payload).__name__}.")
        return state.payload

    @staticmethod
    def _batch_payload(batch: ForwardBatch) -> ARState:
        if not isinstance(batch.payload, ARState):
            raise TypeError(f"Expected ARState batch payload, got {type(batch.payload).__name__}.")
        return batch.payload
