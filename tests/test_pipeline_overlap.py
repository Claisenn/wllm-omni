"""Tests for continuous batching and AR/diffusion pipeline overlap.

Engine level: a request submitted while another is mid-denoise joins the batch
already in flight. Runtime level: the overlapped pipeline starts denoising
before the AR stage has processed every prompt.
"""

from __future__ import annotations

import torch

from wllm_omni.config import EngineConfig
from wllm_omni.engine.diffusion_engine import DiffusionEngine
from wllm_omni.engine.mini_omni_runtime import MiniOmniRuntime
from wllm_omni.engine.model_runner import ModelRunner
from wllm_omni.models.ar_pipeline import ARPipeline, ARTextOutput
from wllm_omni.models.diffusion_executor import DiffusionExecutor
from wllm_omni.request import OmniRequest
from wllm_omni.sampling_params import PRESETS, clone_sampling_params

from wllm_omni.outputs import OmniOutput

from tests.test_step_input_batch import RecordingPipeline


class OutputRecordingPipeline(RecordingPipeline):
    """RecordingPipeline whose decode returns a real OmniOutput."""

    def post_decode(self, state):
        super().post_decode(state)
        return OmniOutput(request_id=state.req_id, frames=[], width=8, height=8, fps=16)


def make_request(prompt: str = "a cat", num_steps: int = 4) -> OmniRequest:
    sampling = clone_sampling_params(PRESETS["quality"])
    sampling.num_inference_steps = num_steps
    return OmniRequest(prompt=prompt, sampling_params=sampling)


def make_engine(pipeline: OutputRecordingPipeline, max_num_seqs: int = 4) -> DiffusionEngine:
    config = EngineConfig(device="cpu", dtype=torch.float32, max_num_seqs=max_num_seqs)
    runner = ModelRunner(config, executors=[DiffusionExecutor(pipeline)])
    return DiffusionEngine(config, runner=runner)


class TestContinuousBatching:
    def test_late_request_joins_the_batch_in_flight(self):
        pipeline = OutputRecordingPipeline()
        engine = make_engine(pipeline)

        first = make_request(num_steps=4)
        engine.submit(first)
        engine.step()
        engine.step()

        # Two steps in, a compatible request arrives and must join mid-flight.
        second = make_request(num_steps=4)
        engine.submit(second)
        outputs = []
        while engine.has_work():
            outputs.extend(engine.step())

        # Rounds: [1, 1] solo, then [2, 2] shared until first finishes,
        # then [1, 1] for the second's remaining steps.
        assert pipeline.denoise_batch_sizes == [1, 1, 2, 2, 1, 1]
        assert {output.request_id for output in outputs} == {first.request_id, second.request_id}

    def test_incompatible_late_request_waits_for_the_batch_to_drain(self):
        pipeline = OutputRecordingPipeline()
        engine = make_engine(pipeline)

        first = make_request(num_steps=3)
        engine.submit(first)
        engine.step()

        incompatible = make_request(num_steps=5)  # different step count -> different key
        engine.submit(incompatible)
        while engine.has_work():
            engine.step()

        # Never batched together: first drains alone, then the other runs alone.
        assert pipeline.denoise_batch_sizes == [1, 1, 1, 1, 1, 1, 1, 1]

    def test_generate_still_drains_everything(self):
        pipeline = OutputRecordingPipeline()
        engine = make_engine(pipeline)
        requests = [make_request(num_steps=2) for _ in range(3)]

        outputs = engine.generate(requests)

        assert {output.request_id for output in outputs} == {request.request_id for request in requests}
        assert pipeline.denoise_batch_sizes == [3, 3]


class SpyARPipeline(ARPipeline):
    """AR stand-in that records the order of AR work relative to denoise work."""

    def __init__(self, event_log: list[str]):
        self.event_log = event_log

    def generate(self, request: OmniRequest) -> ARTextOutput:
        self.event_log.append(f"ar:{request.prompt}")
        return ARTextOutput(request_id=request.request_id, text=request.prompt, tokens=[], token_ids=[])


class LoggingDiffusionPipeline(OutputRecordingPipeline):
    def __init__(self, event_log: list[str]):
        super().__init__()
        self.event_log = event_log

    def denoise_step(self, batch):
        self.event_log.append(f"denoise:{batch.num_reqs}")
        return super().denoise_step(batch)


class TestOverlappedPipeline:
    @staticmethod
    def _make_runtime(event_log: list[str]) -> MiniOmniRuntime:
        config = EngineConfig(enable_mini_omni=True, device="cpu", dtype=torch.float32, max_num_seqs=4)
        runtime = MiniOmniRuntime(config, ar_pipeline=SpyARPipeline(event_log))
        pipeline = LoggingDiffusionPipeline(event_log)
        runner = ModelRunner(config, executors=[DiffusionExecutor(pipeline)])
        runtime.diffusion_stage.engine = DiffusionEngine(config, runner=runner)
        return runtime

    def test_denoising_starts_before_all_prompts_are_rewritten(self):
        event_log: list[str] = []
        runtime = self._make_runtime(event_log)
        requests = [make_request(prompt=f"p{i}", num_steps=4) for i in range(3)]

        runtime.generate_batch_overlapped(requests)

        first_denoise = event_log.index("denoise:1")
        last_ar = len(event_log) - 1 - event_log[::-1].index("ar:p2")
        assert first_denoise < last_ar, (
            f"denoising must begin before the last AR rewrite; log={event_log}"
        )

    def test_batch_grows_as_ar_stage_feeds_requests(self):
        event_log: list[str] = []
        runtime = self._make_runtime(event_log)
        requests = [make_request(prompt=f"p{i}", num_steps=4) for i in range(3)]

        runtime.generate_batch_overlapped(requests)

        denoise_sizes = [int(entry.split(":")[1]) for entry in event_log if entry.startswith("denoise:")]
        # Batch composition ramps up as AR completes each rewrite: 1, then 2,
        # then everyone denoising together until requests drain.
        assert denoise_sizes[0] == 1
        assert denoise_sizes[1] == 2
        assert max(denoise_sizes) == 3

    def test_outputs_return_in_request_order(self):
        event_log: list[str] = []
        runtime = self._make_runtime(event_log)
        requests = [make_request(prompt=f"p{i}", num_steps=2 + i) for i in range(3)]

        outputs = runtime.generate_batch_overlapped(requests)

        assert [output.request_id for output in outputs] == [request.request_id for request in requests]
