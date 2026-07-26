from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch

from wllm_omni.model_types import ModelParadigm
from wllm_omni.models import ModelExecutor, supports_step_execution
from wllm_omni.profiler import RequestProfiler
from wllm_omni.request import OmniRequest
from wllm_omni.sched.interface import StepBatchSamplingParamsKey
from wllm_omni.worker.input_batch import StepInputBatch
from wllm_omni.worker.utils import (
    ExecutionPhase,
    ExecutorCapability,
    ForwardBatch,
    ModelForwardOutput,
    RequestState,
    RunnerOutput,
    RunnerState,
)

if TYPE_CHECKING:
    from wllm_omni.models.wan22 import Wan22I2VPipeline


class DiffusionExecutor(ModelExecutor):
    """Step-wise diffusion executor used by the generic ModelRunner V1.

    The executor owns diffusion-specific state and model calls. The generic
    runner only sees RequestState and ForwardBatch.
    """

    paradigm = ModelParadigm.DIFFUSION
    capabilities = frozenset({
        ExecutorCapability.STEPWISE,
        ExecutorCapability.CACHEABLE_PREPARE,
        ExecutorCapability.MULTIMODAL_INPUT,
    })

    def __init__(self, pipeline: "Wan22I2VPipeline"):
        self.pipeline = pipeline
        if not supports_step_execution(self.pipeline):
            raise TypeError(f"{self.pipeline.__class__.__name__} does not implement the step execution contract.")

    def init_state(self, sched_req_id: str, request: OmniRequest) -> RequestState:
        payload = RunnerState(
            req_id=request.request_id,
            sampling=request.sampling_params,
            prompt=request.prompt,
            image=request.image,
            negative_prompt=request.sampling_params.negative_prompt,
        )
        if self.pipeline.config.enable_profiling:
            payload.extra["profiler"] = RequestProfiler(request.request_id)
        return RequestState(
            req_id=request.request_id,
            sched_req_id=sched_req_id,
            paradigm=self.paradigm,
            payload=payload,
        )

    def batch_key(self, state: RequestState) -> tuple:
        """Group states that may share one forward batch.

        ``ModelRunner._group_states`` groups by this key, so anything
        request-local in here makes batching structurally impossible: the key
        used to carry ``sched_req_id`` and ``step_index``, which guaranteed that
        every request landed in a group of its own. The key now carries exactly
        the batch-compatibility fields, and reuses the scheduler's
        ``StepBatchSamplingParamsKey`` so the two layers cannot drift apart.
        """
        return (self.paradigm.value, StepBatchSamplingParamsKey.from_sampling_params(self._payload(state).sampling))

    def build_forward_batch(self, states: list[RequestState]) -> ForwardBatch:
        payloads = [self._payload(state) for state in states]
        return ForwardBatch(
            paradigm=self.paradigm,
            req_ids=[state.sched_req_id for state in states],
            phase=self._batch_phase(states, payloads),
            payload=payloads,
        )

    @staticmethod
    def _batch_phase(states: list[RequestState], payloads: list[RunnerState]) -> ExecutionPhase:
        """Summarise what the batch is about to do.

        A batch may be mixed: a request admitted this round still needs encoding
        while the others are mid-denoise. The phase reports the earliest work in
        the batch, and is metadata only -- forward() dispatches per request.
        """
        if any(not state.initialized for state in states):
            return ExecutionPhase.PREPARE
        if all(payload.denoise_completed for payload in payloads):
            return ExecutionPhase.FINALIZE
        return ExecutionPhase.STEP

    def forward(self, batch: ForwardBatch) -> ModelForwardOutput:
        if batch.paradigm != self.paradigm:
            raise ValueError(f"DiffusionExecutor cannot run batch for paradigm={batch.paradigm}.")

        payloads = self._batch_payloads(batch)
        anchor = payloads[0]
        outputs: list[RunnerOutput] = []
        with self._profile_stage(anchor, "forward.total"):
            # Requests joining the batch this round are encoded first, one at a
            # time: prepare_encode is per-request work (image preprocessing, VAE
            # encode, prompt encode) and each hits its own caches.
            for payload in payloads:
                if payload.timesteps is None:
                    with self._profile_stage(payload, "forward.prepare_encode"):
                        self.pipeline.prepare_encode(payload)

            active = [payload for payload in payloads if not payload.denoise_completed]
            if active:
                input_batch = StepInputBatch.make_batch(active)
                with self._profile_stage(anchor, "forward.denoise_step"):
                    noise_pred = self.pipeline.denoise_step(input_batch)
                with self._profile_stage(anchor, "forward.step_scheduler"):
                    self.pipeline.step_scheduler(input_batch, noise_pred)

            # Requests reaching their last step decode and finish independently;
            # the rest stay in the batch for the next round.
            for req_id, payload in zip(batch.req_ids, payloads, strict=True):
                if not payload.denoise_completed:
                    outputs.append(RunnerOutput(req_id=req_id, step_index=payload.step_index, finished=False))
                    continue
                with self._profile_stage(payload, "forward.post_decode"):
                    result = self.pipeline.post_decode(payload)
                outputs.append(
                    RunnerOutput(req_id=req_id, step_index=payload.step_index, finished=True, result=result)
                )

        for payload in payloads:
            if payload.denoise_completed and self._profiler(payload) is not None:
                self._emit_profile(payload)
        return ModelForwardOutput(outputs=outputs, payload=payloads)

    def update_states(self, states: list[RequestState], output: ModelForwardOutput) -> None:
        output_by_req_id = {item.req_id: item for item in output.outputs}
        for state in states:
            item = output_by_req_id.get(state.sched_req_id)
            if item is None:
                continue
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
        return output.outputs

    def release(self, state: RequestState) -> None:
        state.payload = None

    @staticmethod
    def _payload(state: RequestState) -> RunnerState:
        if not isinstance(state.payload, RunnerState):
            raise TypeError(f"Expected RunnerState payload, got {type(state.payload).__name__}.")
        return state.payload

    @staticmethod
    def _batch_payloads(batch: ForwardBatch) -> list[RunnerState]:
        if not isinstance(batch.payload, list) or not batch.payload:
            raise TypeError(f"Expected a non-empty RunnerState list payload, got {type(batch.payload).__name__}.")
        for item in batch.payload:
            if not isinstance(item, RunnerState):
                raise TypeError(f"Expected RunnerState payload item, got {type(item).__name__}.")
        return batch.payload

    def _profile_stage(self, state: RunnerState, name: str):
        profile = self._profiler(state)
        if profile is None:
            return nullcontext()
        return profile.stage(name, self._cuda_sync)

    def _profiler(self, state: RunnerState) -> RequestProfiler | None:
        profile = state.extra.get("profiler")
        if profile is None:
            return None
        if not isinstance(profile, RequestProfiler):
            raise TypeError(f"Expected RequestProfiler payload, got {type(profile).__name__}.")
        return profile

    def _emit_profile(self, state: RunnerState) -> None:
        profile = self._profiler(state)
        if profile is None:
            return
        profile.set_metadata(
            steps=state.step_index,
            total_steps=state.total_steps,
            height=state.extra.get("height"),
            width=state.extra.get("width"),
            num_frames=state.extra.get("num_frames"),
            guidance_scale=state.extra.get("guidance_scale"),
            prompt_cache_hit=state.extra.get("prompt_cache_hit"),
            image_cache_hit=state.extra.get("image_cache_hit"),
            condition_cache_hit=state.extra.get("condition_cache_hit"),
            condition_cache_mode=state.extra.get("condition_cache_mode"),
            latents_shape=state.extra.get("latents_shape"),
            condition_shape=state.extra.get("condition_shape"),
            first_frame_mask_shape=state.extra.get("first_frame_mask_shape"),
            denoise_latent_model_input_shape=state.extra.get("denoise_latent_model_input_shape"),
            denoise_timestep_shape=state.extra.get("denoise_timestep_shape"),
            condition_probe_enabled=state.extra.get("condition_probe_enabled"),
            condition_same_across_seed=state.extra.get("condition_same_across_seed"),
            first_frame_mask_same_across_seed=state.extra.get("first_frame_mask_same_across_seed"),
            latents_same_across_seed=state.extra.get("latents_same_across_seed"),
            condition_cache_candidate=state.extra.get("condition_cache_candidate"),
            **self.pipeline.runtime_info(),
        )
        if self.pipeline.config.use_cpu_offload:
            print(
                "[wllm-omni][profile] note cpu_offload=True; timings include CPU/GPU transfer overhead. "
                "Use --disable-cpu-offload for a GPU-resident baseline.",
                flush=True,
            )
        for line in profile.summary_lines():
            print(line, flush=True)

    def _cuda_sync(self) -> None:
        if not torch.cuda.is_available():
            return
        device = getattr(self.pipeline.pipe, "_execution_device", None)
        if device is None:
            return
        device = torch.device(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
