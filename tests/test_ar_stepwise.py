"""Acceptance tests for the AR KV-cache / prefill-decode split (P0).

Two independent guarantees:

1. Fidelity: TransformersARPipeline's prefill/decode_step/finalize loop is
   bit-identical to ``model.generate(do_sample=False, ...)`` on the same
   model -- including when the checkpoint ships a non-default
   generation_config (repetition_penalty, no_repeat_ngram_size, multi-token
   eos lists, min_new_tokens). Uses a tiny randomly-initialized local Qwen2,
   so no network, no downloads, no GPU.
2. Executor contract: ARExecutor advances exactly one step per ``forward()``
   call, carries progress through ModelForwardOutput.payload (surviving a
   serialization boundary, verified via deepcopy), attaches the final result
   inside forward(), falls back to ``generate()`` for pipelines without the
   full stepwise contract, and drops the KV cache as soon as a request
   finishes. The unchanged AREngine/RequestScheduler loop drives multi-step
   decode end to end.
"""

from __future__ import annotations

import copy

import pytest
import torch

from wllm_omni.config import EngineConfig
from wllm_omni.engine.ar_engine import AREngine
from wllm_omni.models import supports_step_execution
from wllm_omni.models.ar_executor import AR_STEP_EXECUTION_METHODS, ARExecutor, ARState
from wllm_omni.models.ar_pipeline import ARTextOutput, IdentityARPipeline, TransformersARPipeline
from wllm_omni.request import OmniRequest
from wllm_omni.worker.utils import ExecutionPhase


# ---------------------------------------------------------------------------
# Part 1: TransformersARPipeline fidelity against model.generate()
# ---------------------------------------------------------------------------

PROMPT_IDS = [3, 8, 5, 7, 9, 4]
MAX_NEW_TOKENS = 16


def _tiny_qwen2_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=2,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


class _FakeBatchEncoding(dict):
    """Minimal BatchEncoding stand-in: dict + attribute access + .to()."""

    def to(self, device):
        return self

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


class _FakeTokenizer:
    """Deterministic tokenizer double emitting a fixed prompt.

    Token-level fidelity is what these tests pin down, so real tokenization
    is irrelevant; a fixed id sequence keeps the reference reproducible.
    """

    chat_template = None
    eos_token_id = 2

    def __init__(self, special_ids=frozenset({1, 2})):
        self.special_ids = set(special_ids)

    def __call__(self, text, return_tensors="pt"):
        ids = list(PROMPT_IDS)
        return _FakeBatchEncoding(
            input_ids=torch.tensor([ids]),
            attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
        )

    def decode(self, ids, skip_special_tokens=True):
        ids = [int(i) for i in ids]
        if skip_special_tokens:
            ids = [i for i in ids if i not in self.special_ids]
        return " ".join(str(i) for i in ids)

    def convert_ids_to_tokens(self, ids):
        return [str(i) for i in ids]


def _make_pipeline(model=None, max_new_tokens: int = MAX_NEW_TOKENS) -> TransformersARPipeline:
    pipeline = TransformersARPipeline.__new__(TransformersARPipeline)
    pipeline.model_path = "tiny/qwen2"
    pipeline.device = torch.device("cpu")
    pipeline.dtype = torch.float32
    pipeline.max_new_tokens = max_new_tokens
    pipeline.tokenizer = _FakeTokenizer()
    pipeline.model = model if model is not None else _tiny_qwen2_model()
    return pipeline


def _reference_generate(pipeline: TransformersARPipeline, max_new_tokens: int = MAX_NEW_TOKENS) -> list[int]:
    """What the pre-P0 monolithic path produced: raw model.generate() ids."""
    inputs = pipeline._tokenize_prompt(pipeline._build_prompt("hello world"))
    with torch.no_grad():
        output_ids = pipeline.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pipeline.tokenizer.eos_token_id,
        )
    return output_ids[0, inputs.input_ids.shape[-1]:].tolist()


def _stepwise_output(pipeline: TransformersARPipeline) -> ARTextOutput:
    """Drive the actual production primitives, not a reimplemented loop."""
    state = pipeline.prefill(OmniRequest(prompt="hello world"))
    steps = 1
    while not state.finished:
        state = pipeline.decode_step(state)
        steps += 1
        assert steps <= pipeline.max_new_tokens, "decode loop must stop at max_new_tokens"
    return pipeline.finalize(state)


def test_stepwise_matches_generate_default_config():
    pipeline = _make_pipeline()
    reference = _reference_generate(pipeline)
    output = _stepwise_output(pipeline)
    assert output.token_ids == reference
    assert output.metadata["token_count"] == len(reference)
    assert output.metadata["input_tokens"] == len(PROMPT_IDS)


def test_stepwise_matches_generate_with_checkpoint_logits_processors():
    """A chat-style checkpoint generation_config (repetition_penalty etc.)
    must flow into the stepwise loop exactly as model.generate() applies it.

    This is the regression gate for the raw-argmax bug: a loop that ignores
    the checkpoint's logits processors diverges here.
    """
    pipeline = _make_pipeline()
    plain_reference = _reference_generate(pipeline)

    pipeline.model.generation_config.repetition_penalty = 1.5
    pipeline.model.generation_config.no_repeat_ngram_size = 2
    penalized_reference = _reference_generate(pipeline)

    # Prove the config actually changes this model's greedy path, so a
    # processor-ignoring implementation cannot pass by accident.
    assert penalized_reference != plain_reference

    output = _stepwise_output(pipeline)
    assert output.token_ids == penalized_reference


def test_stepwise_matches_generate_with_min_new_tokens():
    pipeline = _make_pipeline()
    pipeline.model.generation_config.repetition_penalty = 1.5
    pipeline.model.generation_config.min_new_tokens = 5
    reference = _reference_generate(pipeline)
    assert len(reference) >= 5
    assert _stepwise_output(pipeline).token_ids == reference


def test_eos_stop_triggers_and_matches_generate():
    """EOS must truly end generation early, exactly where generate() ends it,
    with the trailing EOS kept in token_ids (as generate() keeps it) but
    stripped from text via skip_special_tokens."""
    pipeline = _make_pipeline()
    pipeline.model.generation_config.repetition_penalty = 1.5
    pipeline.model.generation_config.no_repeat_ngram_size = 2

    # Pick a token this model actually produces as an additional EOS so the
    # stop path genuinely fires on a randomly-initialized model.
    probe = _reference_generate(pipeline)
    eos_extra = probe[2]
    pipeline.model.generation_config.eos_token_id = [2, eos_extra]
    pipeline.tokenizer.special_ids.add(eos_extra)

    reference = _reference_generate(pipeline)
    assert len(reference) < MAX_NEW_TOKENS, "EOS stop must trigger before the token budget"
    assert reference[-1] == eos_extra

    output = _stepwise_output(pipeline)
    assert output.token_ids == reference
    assert output.token_ids[-1] == eos_extra  # kept in ids, like generate()
    assert str(eos_extra) not in output.text.split()  # stripped from text
    assert output.metadata["token_count"] == len(reference)


def test_max_new_tokens_zero_raises_like_generate():
    """transformers rejects max_new_tokens=0; the stepwise path must surface
    the same validation error rather than silently emitting a token."""
    pipeline = _make_pipeline(max_new_tokens=0)
    with pytest.raises(ValueError, match="max_new_tokens"):
        _reference_generate(pipeline, max_new_tokens=0)
    with pytest.raises(ValueError, match="max_new_tokens"):
        pipeline.prefill(OmniRequest(prompt="hello world"))


def test_pipeline_generate_is_the_stepwise_composition():
    pipeline = _make_pipeline()
    via_generate = pipeline.generate(OmniRequest(prompt="hello world"))
    assert via_generate.token_ids == _reference_generate(pipeline)


# ---------------------------------------------------------------------------
# Part 2: ARExecutor / AREngine stepwise contract (no real model needed)
# ---------------------------------------------------------------------------


class _FakeDecodeState:
    """Opaque decode state double; the executor may only rely on .finished."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.ids = [10]
        self.finished = False
        self.kv_cache = object()  # sentinel standing in for KV tensors


class _FakeStepwisePipeline:
    """Deterministic 3-token stepwise pipeline with full contract."""

    total_tokens = 3

    def __init__(self):
        self.prefill_calls = 0
        self.decode_step_calls = 0

    def prefill(self, request: OmniRequest) -> _FakeDecodeState:
        self.prefill_calls += 1
        return _FakeDecodeState(request.request_id)

    def decode_step(self, state: _FakeDecodeState) -> _FakeDecodeState:
        self.decode_step_calls += 1
        state.ids.append(state.ids[-1] + 1)
        state.finished = len(state.ids) >= self.total_tokens
        return state

    def finalize(self, state: _FakeDecodeState) -> ARTextOutput:
        return ARTextOutput(
            request_id=state.request_id,
            text="-".join(str(i) for i in state.ids),
            tokens=[str(i) for i in state.ids],
            token_ids=list(state.ids),
            metadata={"token_count": len(state.ids)},
        )

    def generate(self, request: OmniRequest) -> ARTextOutput:  # pragma: no cover
        raise AssertionError("generate() must not be called for a stepwise pipeline")


class _PrefillDecodeOnlyPipeline(_FakeStepwisePipeline):
    """Implements prefill/decode_step but NOT finalize; must not be treated
    as stepwise, or the executor would die on the final step."""

    finalize = None

    def generate(self, request: OmniRequest) -> ARTextOutput:
        return ARTextOutput(request_id=request.request_id, text="mono", tokens=["mono"], token_ids=[1])


def test_step_execution_probe_requires_the_full_contract():
    assert AR_STEP_EXECUTION_METHODS == ("prefill", "decode_step", "finalize")
    assert supports_step_execution(_FakeStepwisePipeline(), AR_STEP_EXECUTION_METHODS)
    assert not supports_step_execution(_PrefillDecodeOnlyPipeline(), AR_STEP_EXECUTION_METHODS)
    assert not supports_step_execution(IdentityARPipeline(), AR_STEP_EXECUTION_METHODS)


def test_partial_stepwise_pipeline_falls_back_to_generate():
    executor = ARExecutor(pipeline=_PrefillDecodeOnlyPipeline())
    state = executor.init_state("sched-1", OmniRequest(prompt="x"))
    output = executor.forward(executor.build_forward_batch([state]))
    executor.update_states([state], output)
    assert state.finished is True
    assert output.outputs[0].result.text == "mono"


def test_executor_advances_one_step_per_forward_and_attaches_result():
    pipeline = _FakeStepwisePipeline()
    executor = ARExecutor(pipeline=pipeline)
    state = executor.init_state("sched-1", OmniRequest(prompt="hello"))

    # Step 1: PREPARE (prefill).
    batch = executor.build_forward_batch([state])
    assert batch.phase == ExecutionPhase.PREPARE
    output = executor.forward(batch)
    executor.update_states([state], output)
    assert output.outputs[0].step_index == 1
    assert output.outputs[0].finished is False
    assert output.outputs[0].result is None
    assert state.initialized is True and state.finished is False
    assert pipeline.prefill_calls == 1 and pipeline.decode_step_calls == 0

    # Step 2: STEP (decode), still unfinished.
    batch = executor.build_forward_batch([state])
    assert batch.phase == ExecutionPhase.STEP
    output = executor.forward(batch)
    executor.update_states([state], output)
    assert output.outputs[0].step_index == 2
    assert output.outputs[0].finished is False

    # Step 3: final decode step; result is attached inside forward() and
    # collect_outputs just passes it through.
    output = executor.forward(executor.build_forward_batch([state]))
    executor.update_states([state], output)
    assert output.outputs[0].step_index == 3
    assert output.outputs[0].finished is True
    assert output.outputs[0].result.token_ids == [10, 11, 12]
    assert state.finished is True
    assert executor.collect_outputs([state], output) is output.outputs
    assert pipeline.prefill_calls == 1 and pipeline.decode_step_calls == 2


def test_progress_survives_a_serialization_boundary():
    """Decode progress must round-trip through ModelForwardOutput.payload.

    Deep-copying the batch payload before forward() severs any in-process
    aliasing between the runner's RequestState and the batch -- the situation
    a worker/process boundary creates. An executor that relies on aliasing
    (never reassigning state.payload) re-prefills forever here.
    """
    executor = ARExecutor(pipeline=_FakeStepwisePipeline())
    state = executor.init_state("sched-1", OmniRequest(prompt="hello"))

    for step in range(1, 4):
        batch = executor.build_forward_batch([state])
        batch.payload = copy.deepcopy(batch.payload)
        output = executor.forward(batch)
        executor.update_states([state], output)
        assert state.payload is output.payload
        assert state.step_index == step
        if state.finished:
            break
    else:
        pytest.fail("request did not finish in 3 steps: progress was lost across the boundary")

    assert output.outputs[0].result.token_ids == [10, 11, 12]


def test_kv_cache_is_dropped_as_soon_as_the_request_finishes():
    executor = ARExecutor(pipeline=_FakeStepwisePipeline())
    state = executor.init_state("sched-1", OmniRequest(prompt="hello"))
    while not state.finished:
        output = executor.forward(executor.build_forward_batch([state]))
        executor.update_states([state], output)
    payload = state.payload
    assert isinstance(payload, ARState)
    assert payload.decode is None, "decode state (KV cache) must be freed at finalize"
    assert payload.output is not None
    executor.release(state)
    assert state.payload is None


def test_identity_pipeline_single_shot_behavior_unchanged():
    executor = ARExecutor(pipeline=IdentityARPipeline())
    assert executor._stepwise is False
    state = executor.init_state("sched-1", OmniRequest(prompt="a cat wearing sunglasses"))

    batch = executor.build_forward_batch([state])
    assert batch.phase == ExecutionPhase.FINALIZE
    output = executor.forward(batch)
    executor.update_states([state], output)

    item = output.outputs[0]
    assert item.finished is True and item.step_index == 1
    assert item.result.metadata["mode"] == "identity_prompt_bridge"
    assert state.finished is True


def test_ar_engine_drives_multi_step_decode_with_unchanged_scheduler():
    """AREngine/RequestScheduler need zero changes: their per-step loop keeps
    re-scheduling running requests until the executor reports finished."""
    pipeline = _FakeStepwisePipeline()
    engine = AREngine(EngineConfig(enable_mini_omni=True), pipeline=pipeline)

    output = engine.generate(OmniRequest(prompt="hello"))

    assert output.token_ids == [10, 11, 12]
    assert pipeline.prefill_calls == 1
    assert pipeline.decode_step_calls == 2  # 3 forward() calls: prefill + 2 decode steps
    assert not engine.runner.state_cache, "finished request state must be released"


def test_ar_engine_still_works_with_identity_pipeline():
    engine = AREngine(EngineConfig(enable_mini_omni=True))  # defaults to IdentityARPipeline
    output = engine.generate(OmniRequest(prompt="a cat wearing sunglasses"))
    assert output.metadata["mode"] == "identity_prompt_bridge"
    assert output.text == "a cat wearing sunglasses"


def test_transformers_pipeline_end_to_end_through_engine():
    """The real stepwise pipeline driven by the real engine loop must equal
    model.generate() on a checkpoint with a non-default generation_config."""
    pipeline = _make_pipeline()
    pipeline.model.generation_config.repetition_penalty = 1.5
    reference = _reference_generate(pipeline)

    engine = AREngine(EngineConfig(enable_mini_omni=True), pipeline=pipeline)
    output = engine.generate(OmniRequest(prompt="hello world"))
    assert output.token_ids == reference
