from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from wllm_omni.model_types import ModelParadigm
from wllm_omni.models import ModelExecutor, supports_step_execution
from wllm_omni.models.ar_pipeline import ARDecodeState, ARPipeline, ARTextOutput, IdentityARPipeline
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

AR_STEP_EXECUTION_METHODS = ("prefill", "decode_step")


@dataclass(slots=True)
class ARState:
    request: OmniRequest
    decode: ARDecodeState | None = None
    output: ARTextOutput | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class ARExecutor(ModelExecutor):
    """AR executor for the mini-Omni runtime.

    When the configured pipeline implements the stepwise contract
    (``prefill``/``decode_step``, e.g. TransformersARPipeline), each call to
    ``forward`` advances exactly one decode step and the KV cache lives in
    ``ARState.decode`` across ``ModelRunner.execute()`` calls -- the same
    PREPARE-then-STEP shape DiffusionExecutor uses for denoise steps, just
    driven by token count instead of a fixed step count. Pipelines that only
    implement ``generate`` (e.g. IdentityARPipeline) still work: forward()
    falls back to running them to completion in a single call.
    """

    paradigm = ModelParadigm.AUTOREGRESSIVE
    capabilities = frozenset(
        {ExecutorCapability.STEPWISE, ExecutorCapability.STREAMING, ExecutorCapability.KV_CACHE}
    )

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
        return (self.paradigm.value,)

    def build_forward_batch(self, states: list[RequestState]) -> ForwardBatch:
        return ForwardBatch(
            paradigm=self.paradigm,
            req_ids=[state.sched_req_id for state in states],
            phase=ExecutionPhase.STEP,
            payload=[self._state_payload(state) for state in states],
        )

    def forward(self, batch: ForwardBatch) -> ModelForwardOutput:
        if batch.paradigm != self.paradigm:
            raise ValueError(f"ARExecutor cannot run batch for paradigm={batch.paradigm}.")
        states = self._batch_payload(batch)
        outputs: list[RunnerOutput] = []
        for req_id, ar_state in zip(batch.req_ids, states, strict=True):
            self._advance(ar_state)
            step_index = len(ar_state.decode.generated_ids) if ar_state.decode is not None else 1
            outputs.append(RunnerOutput(req_id=req_id, step_index=step_index, finished=ar_state.output is not None))
        return ModelForwardOutput(outputs=outputs, payload=states)

    def _advance(self, ar_state: ARState) -> None:
        """Run one prefill/decode step, or the whole request in one shot for
        pipelines that don't support stepwise decoding."""
        if not self._stepwise:
            ar_state.output = self.pipeline.generate(ar_state.request)
            return
        if ar_state.decode is None:
            ar_state.decode = self.pipeline.prefill(ar_state.request)
        else:
            ar_state.decode = self.pipeline.decode_step(ar_state.decode)
        if ar_state.decode.finished:
            ar_state.output = self.pipeline.finalize(ar_state.decode)

    def update_states(self, states: list[RequestState], output: ModelForwardOutput) -> None:
        output_by_req_id = {item.req_id: item for item in output.outputs}
        for state in states:
            item = output_by_req_id.get(state.sched_req_id)
            if item is None:
                continue
            state.initialized = True
            state.step_index = item.step_index or state.step_index
            state.error = item.error
            state.finished = item.finished

    def collect_outputs(
        self,
        states: list[RequestState],
        output: ModelForwardOutput,
    ) -> list[RunnerOutput]:
        results: list[RunnerOutput] = []
        output_by_req_id = {item.req_id: item for item in output.outputs}
        for state in states:
            item = output_by_req_id.get(state.sched_req_id)
            if item is None:
                continue
            payload = self._state_payload(state)
            results.append(
                RunnerOutput(
                    req_id=state.sched_req_id,
                    step_index=item.step_index,
                    finished=item.finished,
                    result=payload.output,
                    error=item.error,
                )
            )
        return results

    def release(self, state: RequestState) -> None:
        state.payload = None

    @staticmethod
    def _state_payload(state: RequestState) -> ARState:
        if not isinstance(state.payload, ARState):
            raise TypeError(f"Expected ARState payload, got {type(state.payload).__name__}.")
        return state.payload

    @staticmethod
    def _batch_payload(batch: ForwardBatch) -> list[ARState]:
        if not isinstance(batch.payload, list):
            raise TypeError(f"Expected ARState list payload, got {type(batch.payload).__name__}.")
        for item in batch.payload:
            if not isinstance(item, ARState):
                raise TypeError(f"Expected ARState payload item, got {type(item).__name__}.")
        return batch.payload
