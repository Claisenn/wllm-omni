from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from wllm_omni.config import EngineConfig
from wllm_omni.engine.ar_engine import AREngine
from wllm_omni.engine.diffusion_engine import DiffusionEngine
from wllm_omni.model_types import ModelParadigm
from wllm_omni.models.ar_pipeline import ARPipeline, ARTextOutput
from wllm_omni.outputs import OmniOutput
from wllm_omni.request import OmniRequest


@dataclass(slots=True)
class StageOutput:
    request_id: str
    data: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class Stage(ABC):
    name: str
    paradigm: ModelParadigm

    def prepare(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    def run(self, request: OmniRequest) -> StageOutput:
        pass

    def run_batch(self, requests: list[OmniRequest]) -> list[StageOutput]:
        """Run several requests through this stage.

        The default is a per-request loop, which is correct for any stage. A
        stage whose engine can exploit request-level batching (the diffusion
        stage: compatible requests share one denoise-step forward) overrides
        this to hand the whole set over at once.
        """
        return [self.run(request) for request in requests]


class ARStage(Stage):
    name = "ar.prompt_bridge"
    paradigm = ModelParadigm.AUTOREGRESSIVE

    def __init__(self, config: EngineConfig, pipeline: ARPipeline | None = None):
        self.engine = AREngine(config, pipeline=pipeline)

    def run(self, request: OmniRequest) -> StageOutput:
        ar_output = self.engine.generate(request)
        return StageOutput(
            request_id=request.request_id,
            data=ar_output,
            metadata={
                "mode": ar_output.metadata.get("mode"),
                "model": ar_output.metadata.get("model"),
                "input_tokens": ar_output.metadata.get("input_tokens"),
                "output_tokens": ar_output.metadata.get("token_count", len(ar_output.tokens)),
            },
        )


class DiffusionStage(Stage):
    name = "diffusion.wan22_i2v"
    paradigm = ModelParadigm.DIFFUSION

    def __init__(self, config: EngineConfig):
        self.config = config
        self.engine: DiffusionEngine | None = None

    def prepare(self) -> dict[str, Any]:
        if self.engine is not None:
            return {"load_elapsed_s": 0.0, "load_was_cold": False}
        start = perf_counter()
        self.engine = DiffusionEngine(self.config)
        return {"load_elapsed_s": perf_counter() - start, "load_was_cold": True}

    def run(self, request: OmniRequest) -> StageOutput:
        outputs = self._engine().generate(request)
        if not outputs:
            raise RuntimeError("Diffusion stage finished without output.")
        return StageOutput(
            request_id=request.request_id,
            data=outputs[0],
            metadata={
                "bridge": "ar_text_to_diffusion_prompt",
            },
        )

    def run_batch(self, requests: list[OmniRequest]) -> list[StageOutput]:
        """Submit all requests to the diffusion engine in one call.

        This is where the omni pipeline meets denoise-step batching: the
        engine's scheduler admits every compatible request into one running
        set, so N videos cost one denoise loop instead of N. The engine
        reports completions in finish order; outputs are matched back to
        requests by request_id.
        """
        outputs = self._engine().generate(list(requests))
        outputs_by_id = {output.request_id: output for output in outputs}
        missing = [request.request_id for request in requests if request.request_id not in outputs_by_id]
        if missing:
            raise RuntimeError(f"Diffusion stage finished without output for requests: {missing}.")
        return [
            StageOutput(
                request_id=request.request_id,
                data=outputs_by_id[request.request_id],
                metadata={
                    "bridge": "ar_text_to_diffusion_prompt",
                },
            )
            for request in requests
        ]

    def _engine(self) -> DiffusionEngine:
        if self.engine is None:
            self.engine = DiffusionEngine(self.config)
        return self.engine
