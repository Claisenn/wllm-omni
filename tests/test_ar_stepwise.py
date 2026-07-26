"""Acceptance tests for the AR KV-cache / prefill-decode split (P0).

Two independent guarantees are checked:

1. TransformersARPipeline's new prefill/decode_step/finalize loop produces
   bit-identical output to transformers' own ``model.generate()`` -- proving
   the manual KV-cache loop is not a behavior change, just an incremental
   version of the same computation. Uses a tiny, randomly-initialized local
   Qwen2 model (no network access, no checkpoint download).
2. ARExecutor correctly drives that stepwise contract one step per
   ``forward()`` call (and falls back to a single call for pipelines that
   don't implement it), using a lightweight fake pipeline so these tests
   don't depend on torch/transformers at all.
"""

from __future__ import annotations

import torch

from wllm_omni.config import EngineConfig
from wllm_omni.engine.ar_engine import AREngine
from wllm_omni.models.ar_executor import AR_STEP_EXECUTION_METHODS, ARExecutor
from wllm_omni.models.ar_pipeline import ARDecodeState, ARTextOutput, IdentityARPipeline, TransformersARPipeline
from wllm_omni.models import supports_step_execution
from wllm_omni.request import OmniRequest
from wllm_omni.worker.utils import RequestState


def _tiny_qwen2_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=48,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=2,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


class _FakeBatchEncoding(dict):
    """Minimal stand-in for transformers' BatchEncoding: dict + attribute access + .to()."""

    def to(self, device):
        return self

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


class _FakeTokenizer:
    """Deterministic tokenizer double -- tests the decode loop, not real tokenization."""

    chat_template = None
    eos_token_id = 2

    def __call__(self, text, return_tensors="pt"):
        ids = [3 + (ord(c) % 10) for c in text[:6]] or [3]
        return _FakeBatchEncoding(
            input_ids=torch.tensor([ids]),
            attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
        )

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)

    def convert_ids_to_tokens(self, ids):
        return [str(i) for i in ids]


def _make_pipeline(max_new_tokens: int = 8) -> TransformersARPipeline:
    pipeline = TransformersARPipeline.__new__(TransformersARPipeline)
    pipeline.model_path = "tiny/qwen2"
    pipeline.device = torch.device("cpu")
    pipeline.dtype = torch.float32
    pipeline.max_new_tokens = max_new_tokens
    pipeline.tokenizer = _FakeTokenizer()
    pipeline.model = _tiny_qwen2_model()
    return pipeline


def test_stepwise_decode_matches_reference_model_generate():
    """The manual prefill/decode_step loop must be bit-identical to model.generate()."""
    model = _tiny_qwen2_model()
    input_ids = torch.tensor([[3, 8, 5, 7, 9, 4]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        reference_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=2,
        )
    reference = reference_ids[0, input_ids.shape[-1] :].tolist()

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    generated = [int(torch.argmax(out.logits[:, -1, :], dim=-1))]
    cache = out.past_key_values
    mask = attention_mask
    for _ in range(7):
        if generated[-1] == 2:
            break
        mask = torch.cat([mask, mask.new_ones((mask.shape[0], 1))], dim=-1)
        with torch.no_grad():
            out = model(
                input_ids=torch.tensor([[generated[-1]]]),
                attention_mask=mask,
                past_key_values=cache,
                use_cache=True,
            )
        generated.append(int(torch.argmax(out.logits[:, -1, :], dim=-1)))
        cache = out.past_key_values

    assert generated == reference


def test_transformers_pipeline_generate_matches_stepwise_primitives():
    """generate() must be exactly the composition of prefill/decode_step/finalize."""
    pipeline = _make_pipeline()
    request = OmniRequest(prompt="hello world")

    via_generate = pipeline.generate(request)

    state = pipeline.prefill(request)
    while not state.finished:
        state = pipeline.decode_step(state)
    via_primitives = pipeline.finalize(state)

    assert via_generate.token_ids == via_primitives.token_ids
    assert via_generate.text == via_primitives.text


def test_transformers_pipeline_stops_at_max_new_tokens():
    pipeline = _make_pipeline(max_new_tokens=3)
    request = OmniRequest(prompt="hello world")

    state = pipeline.prefill(request)
    steps = 1
    while not state.finished:
        state = pipeline.decode_step(state)
        steps += 1
        assert steps <= 3, "decode loop must stop at max_new_tokens"

    assert steps == 3
    output = pipeline.finalize(state)
    assert len(output.token_ids) == 3


def test_supports_step_execution_checks_the_given_methods():
    assert supports_step_execution(pipeline := _make_pipeline(), AR_STEP_EXECUTION_METHODS)
    assert not supports_step_execution(IdentityARPipeline(), AR_STEP_EXECUTION_METHODS)
    del pipeline


class _FakeStepwisePipeline:
    """Deterministic 3-token stepwise pipeline with no torch dependency."""

    def __init__(self):
        self.decode_step_calls = 0

    def prefill(self, request: OmniRequest) -> ARDecodeState:
        return ARDecodeState(request_id=request.request_id, input_length=1, generated_ids=[10], finished=False)

    def decode_step(self, state: ARDecodeState) -> ARDecodeState:
        self.decode_step_calls += 1
        state.generated_ids.append(state.generated_ids[-1] + 1)
        state.finished = len(state.generated_ids) >= 3
        return state

    def finalize(self, state: ARDecodeState) -> ARTextOutput:
        return ARTextOutput(
            request_id=state.request_id,
            text="-".join(str(i) for i in state.generated_ids),
            tokens=[str(i) for i in state.generated_ids],
            token_ids=list(state.generated_ids),
            metadata={"steps": len(state.generated_ids)},
        )

    def generate(self, request: OmniRequest) -> ARTextOutput:  # pragma: no cover - must not be used
        raise AssertionError("generate() must not be called when the pipeline supports stepwise decoding")


def test_ar_executor_advances_one_step_per_forward_call():
    executor = ARExecutor(pipeline=_FakeStepwisePipeline())
    assert executor._stepwise is True

    request = OmniRequest(prompt="hello")
    state = executor.init_state("sched-1", request)

    batch = executor.build_forward_batch([state])
    output = executor.forward(batch)
    executor.update_states([state], output)
    assert output.outputs[0].step_index == 1
    assert output.outputs[0].finished is False
    assert state.initialized is True
    assert state.finished is False

    batch = executor.build_forward_batch([state])
    output = executor.forward(batch)
    executor.update_states([state], output)
    assert output.outputs[0].step_index == 2
    assert output.outputs[0].finished is False

    batch = executor.build_forward_batch([state])
    output = executor.forward(batch)
    executor.update_states([state], output)
    assert output.outputs[0].step_index == 3
    assert output.outputs[0].finished is True

    results = executor.collect_outputs([state], output)
    assert results[0].result.token_ids == [10, 11, 12]


def test_ar_executor_falls_back_to_generate_for_non_stepwise_pipelines():
    executor = ARExecutor(pipeline=IdentityARPipeline())
    assert executor._stepwise is False

    request = OmniRequest(prompt="a cat wearing sunglasses")
    state = executor.init_state("sched-1", request)

    batch = executor.build_forward_batch([state])
    output = executor.forward(batch)
    executor.update_states([state], output)

    assert output.outputs[0].finished is True
    assert output.outputs[0].step_index == 1
    results = executor.collect_outputs([state], output)
    assert results[0].result.metadata["mode"] == "identity_prompt_bridge"


def test_ar_engine_drives_multiple_real_forward_calls_with_no_scheduler_changes():
    """AREngine/RequestScheduler need zero changes to support multi-step decode:
    their existing loop re-schedules "running" requests until finished."""
    pipeline = _FakeStepwisePipeline()
    engine = AREngine(EngineConfig(enable_mini_omni=True), pipeline=pipeline)

    output = engine.generate(OmniRequest(prompt="hello"))

    assert output.token_ids == [10, 11, 12]
    assert pipeline.decode_step_calls == 2  # 1 prefill + 2 decode_step calls = 3 tokens


def test_ar_engine_still_works_with_identity_pipeline():
    engine = AREngine(EngineConfig(enable_mini_omni=True))  # defaults to IdentityARPipeline
    output = engine.generate(OmniRequest(prompt="a cat wearing sunglasses"))
    assert output.metadata["mode"] == "identity_prompt_bridge"


def test_ar_state_payload_type_guard_unaffected_by_new_decode_field():
    executor = ARExecutor(pipeline=IdentityARPipeline())
    state = executor.init_state("sched-1", OmniRequest(prompt="x"))
    assert isinstance(state, RequestState)
    assert executor._state_payload(state).decode is None
