"""Tests for running multiple multimodal requests through the AR -> diffusion pipeline.

This is where the omni orchestration meets denoise-step batching: the AR stage
rewrites each prompt per request, the connector bridges each AR text into a
diffusion request, and the diffusion stage hands the whole set to its engine in
one call so compatible requests share a denoise loop.
"""

from __future__ import annotations

import torch

from wllm_omni.config import EngineConfig
from wllm_omni.engine.mini_omni_runtime import MiniOmniRuntime
from wllm_omni.models.ar_pipeline import IdentityARPipeline
from wllm_omni.outputs import OmniOutput
from wllm_omni.request import OmniRequest
from wllm_omni.sampling_params import PRESETS, clone_sampling_params


class FakeDiffusionEngine:
    """Records generate() calls; returns one OmniOutput per request."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def generate(self, requests):
        if not isinstance(requests, list):
            requests = [requests]
        self.calls.append([request.request_id for request in requests])
        # Return in reverse order to prove callers reorder by request_id.
        return [
            OmniOutput(request_id=request.request_id, frames=[], width=576, height=800, fps=16)
            for request in reversed(requests)
        ]


def make_runtime() -> tuple[MiniOmniRuntime, FakeDiffusionEngine]:
    config = EngineConfig(enable_mini_omni=True, device="cpu", dtype=torch.float32)
    runtime = MiniOmniRuntime(config, ar_pipeline=IdentityARPipeline())
    engine = FakeDiffusionEngine()
    runtime.diffusion_stage.engine = engine
    return runtime, engine


def make_request(prompt: str) -> OmniRequest:
    return OmniRequest(prompt=prompt, sampling_params=clone_sampling_params(PRESETS["quality"]))


class TestOmniBatchPipeline:
    def test_diffusion_stage_receives_the_whole_batch_in_one_call(self):
        runtime, engine = make_runtime()
        requests = [make_request(f"a cat, take {i}") for i in range(3)]

        runtime.generate_batch(requests)

        assert len(engine.calls) == 1, "diffusion must see the batch once, not once per request"
        assert engine.calls[0] == [request.request_id for request in requests]

    def test_outputs_come_back_in_request_order(self):
        runtime, engine = make_runtime()
        requests = [make_request(f"scene {i}") for i in range(3)]

        outputs = runtime.generate_batch(requests)

        # FakeDiffusionEngine returns them reversed; the runtime must restore order.
        assert [output.request_id for output in outputs] == [request.request_id for request in requests]

    def test_ar_stage_rewrites_each_prompt_individually(self):
        runtime, _ = make_runtime()
        requests = [make_request("  spaced   out  prompt  "), make_request("another one")]

        runtime.generate_batch(requests)

        ar_records = [record for record in runtime.last_trace.stages if record.name == "ar.prompt_bridge"]
        assert len(ar_records) == 2
        assert {record.request_id for record in ar_records} == {request.request_id for request in requests}

    def test_trace_covers_every_request_and_stage(self):
        runtime, _ = make_runtime()
        requests = [make_request("one"), make_request("two")]

        runtime.generate_batch(requests)

        trace = runtime.last_trace
        # 2 stages x 2 requests = 4 records, each tagged with the batch size.
        assert len(trace.stages) == 4
        assert all(record.metadata.get("batch_size") == 2 for record in trace.stages)

    def test_single_request_batch_matches_single_request_path(self):
        runtime, engine = make_runtime()
        request = make_request("solo")

        outputs = runtime.generate_batch([request])

        assert len(outputs) == 1
        assert outputs[0].request_id == request.request_id
        assert engine.calls == [[request.request_id]]

    def test_connector_bridges_ar_text_into_each_diffusion_request(self):
        runtime, _ = make_runtime()
        captured: list[str] = []
        original_connect = runtime.connector.connect

        def spy(context):
            request = original_connect(context)
            captured.append(request.prompt)
            return request

        runtime.connector.connect = spy
        # Rebuild the graph so the edge holds the spying connector.
        runtime.graph = runtime._build_default_graph()
        runtime.stage_scheduler = type(runtime.stage_scheduler)(runtime.graph)

        runtime.generate_batch([make_request("  a   dog  "), make_request("a bird")])

        # IdentityARPipeline normalizes whitespace; the bridged prompts must be
        # the per-request AR outputs, not a shared one.
        assert captured == ["a dog", "a bird"]
