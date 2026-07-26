"""Tests for batched classifier-free guidance row helpers.

The core claim is mathematical equivalence: running one merged 2N-row forward
and splitting must give exactly what two sequential N-row forwards give, for
any per-row deterministic transformer. The equivalence is exact on row level;
on real GPUs the batched matmul may reduce in a different order, which is why
the feature stays opt-in and the GPU comparison uses a tolerance.
"""

from __future__ import annotations

import pytest
import torch

from wllm_omni.config import EngineConfig
from wllm_omni.worker.cfg import merge_cfg_rows, split_cfg_rows

N, C, F, H, W = 2, 4, 3, 8, 8
SEQ, DIM = 6, 16


def make_inputs():
    torch.manual_seed(0)
    latent = torch.randn(N, C, F, H, W)
    timestep = torch.randn(N)
    prompt = torch.randn(N, SEQ, DIM)
    negative = torch.randn(N, SEQ, DIM)
    return latent, timestep, prompt, negative


def fake_transformer(hidden: torch.Tensor, timestep: torch.Tensor, encoder: torch.Tensor) -> torch.Tensor:
    """Deterministic per-row function standing in for the DiT.

    Row i of the output depends only on row i of each input, which is the
    property the real transformer has along the batch dimension.
    """
    scale = encoder.mean(dim=(1, 2)) + timestep
    return hidden * scale.view(-1, 1, 1, 1, 1)


class TestMergeSplit:
    def test_merged_forward_equals_two_sequential_forwards(self):
        latent, timestep, prompt, negative = make_inputs()

        cond_seq = fake_transformer(latent, timestep, prompt)
        uncond_seq = fake_transformer(latent, timestep, negative)

        hidden, timesteps, encoder = merge_cfg_rows(latent, timestep, prompt, negative)
        cond_merged, uncond_merged = split_cfg_rows(fake_transformer(hidden, timesteps, encoder), N)

        assert torch.equal(cond_merged, cond_seq)
        assert torch.equal(uncond_merged, uncond_seq)

    def test_cfg_combine_is_identical_between_paths(self):
        latent, timestep, prompt, negative = make_inputs()
        scale = 5.0

        cond = fake_transformer(latent, timestep, prompt)
        uncond = fake_transformer(latent, timestep, negative)
        sequential = uncond + scale * (cond - uncond)

        hidden, timesteps, encoder = merge_cfg_rows(latent, timestep, prompt, negative)
        cond_m, uncond_m = split_cfg_rows(fake_transformer(hidden, timesteps, encoder), N)
        merged = uncond_m + scale * (cond_m - uncond_m)

        assert torch.equal(merged, sequential)

    def test_merge_handles_expand_timesteps_shape(self):
        """TI2V-5B expands timesteps to (N, patches); merging must repeat rows."""
        latent, _, prompt, negative = make_inputs()
        timestep = torch.randn(N, 12)

        _, timesteps, _ = merge_cfg_rows(latent, timestep, prompt, negative)

        assert timesteps.shape == (2 * N, 12)
        assert torch.equal(timesteps[:N], timestep)
        assert torch.equal(timesteps[N:], timestep)

    def test_mismatched_embedding_shapes_are_rejected(self):
        latent, timestep, prompt, _ = make_inputs()
        short_negative = torch.randn(N, SEQ - 2, DIM)

        with pytest.raises(ValueError, match="equal shape"):
            merge_cfg_rows(latent, timestep, prompt, short_negative)

    def test_split_rejects_wrong_row_count(self):
        with pytest.raises(ValueError, match="rows"):
            split_cfg_rows(torch.randn(3, 1), num_rows=2)


class TestConfigDefault:
    def test_cfg_batched_is_opt_in(self):
        assert EngineConfig(device="cpu", dtype=torch.float32).cfg_batched is False
