"""Phase-1 gate: the P0 stepwise AR contract holds on an MLA-architecture model.

FlashMLA (the eventual Phase-2 target) only accelerates MLA attention, so
before any kernel work the pipeline must be proven correct on a DeepSeek-style
model. These tests mirror the Qwen2 fidelity suite on a tiny
randomly-initialized DeepseekV3 (real MLA geometry: latent q/kv compression,
rope/nope head split, small MoE) -- no network, no downloads, no GPU.

They also pin the baseline gap FlashMLA exists to close: transformers'
DeepseekV3 implementation materializes per-head K/V into the cache instead of
storing the compressed latent, so the MLA memory advantage is NOT realized by
the stock HF path.
"""

from __future__ import annotations

import pytest
import torch

from wllm_omni.config import EngineConfig
from wllm_omni.engine.ar_engine import AREngine
from wllm_omni.models.ar_pipeline import TransformersARPipeline
from wllm_omni.request import OmniRequest

from tests.test_ar_stepwise import _FakeTokenizer, _reference_generate, _stepwise_output

PROMPT_LEN = 6
MLA_GEOMETRY = dict(
    q_lora_rank=16,
    kv_lora_rank=16,
    qk_rope_head_dim=8,
    qk_nope_head_dim=8,
    v_head_dim=8,
)


def _tiny_deepseek_v3_model():
    from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

    torch.manual_seed(0)
    config = DeepseekV3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        first_k_dense_replace=1,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=2,
        **MLA_GEOMETRY,
    )
    model = DeepseekV3ForCausalLM(config)
    model.eval()
    return model


def _make_mla_pipeline(max_new_tokens: int = 12) -> TransformersARPipeline:
    pipeline = TransformersARPipeline.__new__(TransformersARPipeline)
    pipeline.model_path = "tiny/deepseek-v3"
    pipeline.device = torch.device("cpu")
    pipeline.dtype = torch.float32
    pipeline.max_new_tokens = max_new_tokens
    pipeline.tokenizer = _FakeTokenizer()
    pipeline.model = _tiny_deepseek_v3_model()
    return pipeline


def test_stepwise_matches_generate_on_mla_architecture():
    """The P0 contract must hold unchanged on MLA attention: prefill /
    decode_step / finalize bit-identical to model.generate()."""
    pipeline = _make_mla_pipeline()
    reference = _reference_generate(pipeline, max_new_tokens=12)
    output = _stepwise_output(pipeline)
    assert output.token_ids == reference


def test_stepwise_matches_generate_on_mla_with_checkpoint_logits_processors():
    """Same regression gate as the Qwen2 suite, on MLA: a chat-style
    generation_config must flow through identically."""
    pipeline = _make_mla_pipeline()
    plain = _reference_generate(pipeline, max_new_tokens=12)

    pipeline.model.generation_config.repetition_penalty = 1.5
    pipeline.model.generation_config.no_repeat_ngram_size = 2
    penalized = _reference_generate(pipeline, max_new_tokens=12)
    assert penalized != plain, "config must actually change this model's greedy path"

    assert _stepwise_output(pipeline).token_ids == penalized


def test_hf_deepseek_cache_materializes_per_head_kv():
    """Baseline-gap probe for Phase 2.

    True MLA caching stores one compressed latent per token
    (kv_lora_rank + qk_rope_head_dim floats, shared across heads).
    transformers' DeepseekV3 instead caches decompressed per-head K
    (qk_nope + qk_rope dims) and V (v_head_dim) -- so the stock HF path
    scales with num_heads and does NOT realize the MLA memory saving.
    FlashMLA integration (Phase 2) is what would close this gap; if an HF
    upgrade ever makes this test fail, the baseline has changed and the
    Phase-2 benchmark comparison must be re-established.
    """
    model = _tiny_deepseek_v3_model()
    input_ids = torch.tensor([[3, 8, 5, 7, 9, 4]])
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=True)

    layer0 = out.past_key_values.layers[0]
    num_heads = model.config.num_attention_heads
    qk_head_dim = MLA_GEOMETRY["qk_nope_head_dim"] + MLA_GEOMETRY["qk_rope_head_dim"]

    assert layer0.keys.shape == (1, num_heads, PROMPT_LEN, qk_head_dim)
    assert layer0.values.shape == (1, num_heads, PROMPT_LEN, MLA_GEOMETRY["v_head_dim"])

    # What a latent cache would cost vs. what HF actually stores, per token:
    latent_floats = MLA_GEOMETRY["kv_lora_rank"] + MLA_GEOMETRY["qk_rope_head_dim"]
    materialized_floats = num_heads * (qk_head_dim + MLA_GEOMETRY["v_head_dim"])
    assert materialized_floats > latent_floats


def test_mla_pipeline_end_to_end_through_engine():
    """The unchanged AREngine loop drives the MLA backend stepwise."""
    pipeline = _make_mla_pipeline()
    reference = _reference_generate(pipeline, max_new_tokens=12)

    engine = AREngine(EngineConfig(enable_mini_omni=True), pipeline=pipeline)
    output = engine.generate(OmniRequest(prompt="hello world"))
    assert output.token_ids == reference
    assert not engine.runner.state_cache
