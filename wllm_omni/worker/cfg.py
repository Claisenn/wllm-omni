"""Row-level helpers for batched classifier-free guidance.

CFG needs two transformer evaluations per denoise step: one conditioned on the
prompt embeddings, one on the negative embeddings. Sequentially that is two
forwards over N rows each. These helpers express the alternative: stack the two
branches along the batch dimension -- the same row-concatenation StepInputBatch
uses across requests, applied within a request -- so a single forward over 2N
rows serves both, halving kernel launches and weight reads per step.

The trade-off is activation memory: the merged forward holds 2N rows of
activations at once, which is why the caller keeps this behind a config switch
(EngineConfig.cfg_batched) instead of making it the default.
"""

from __future__ import annotations

import torch


def merge_cfg_rows(
    latent_model_input: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack the cond and uncond branches into one 2N-row batch.

    The latent input and timestep are identical for both branches -- only the
    encoder states differ -- so they are repeated, while the embeddings are
    concatenated cond-first. Wan pads prompt and negative embeddings to the
    same sequence length (max_sequence_length=512), which is what makes the
    concat legal.
    """
    if prompt_embeds.shape != negative_prompt_embeds.shape:
        raise ValueError(
            "Batched CFG needs prompt and negative embeddings of equal shape, "
            f"got {tuple(prompt_embeds.shape)} vs {tuple(negative_prompt_embeds.shape)}."
        )
    hidden_states = torch.cat([latent_model_input, latent_model_input], dim=0)
    timesteps = torch.cat([timestep, timestep], dim=0)
    encoder_hidden_states = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
    return hidden_states, timesteps, encoder_hidden_states


def split_cfg_rows(noise_pred: torch.Tensor, num_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a merged 2N-row prediction back into (cond, uncond)."""
    if noise_pred.shape[0] != 2 * num_rows:
        raise ValueError(
            f"Expected a merged prediction with {2 * num_rows} rows, got {noise_pred.shape[0]}."
        )
    return noise_pred[:num_rows], noise_pred[num_rows:]
