"""Step-level batch assembly for diffusion requests.

The split mirrors ``vllm_omni.diffusion.worker.input_batch``: :class:`RunnerState`
is the persistent per-request source of truth, and :class:`StepInputBatch` is an
ephemeral contiguous view over the requests scheduled for the current denoise
step. Running a step means gathering rows out of the request states, calling the
transformer once, and scattering the result back.

Requests in one batch are guaranteed by
:class:`~wllm_omni.sched.interface.StepBatchSamplingParamsKey` to agree on
resolution, frame count and guidance scale, so every gathered tensor stacks
without padding. They are *not* required to agree on progress: each request
carries its own timestep schedule and its own scheduler instance, so a request
admitted mid-flight simply contributes a different timestep row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from wllm_omni.worker.utils import RunnerState


def _require(states: Sequence[RunnerState], attr: str) -> list[torch.Tensor]:
    values = []
    for state in states:
        value = getattr(state, attr, None)
        if value is None:
            raise ValueError(f"Request {state.req_id} is missing {attr!r}; prepare_encode must run first.")
        values.append(value)
    return values


def _stack(tensors: Sequence[torch.Tensor], name: str) -> torch.Tensor:
    """Concatenate per-request tensors along the batch dimension.

    Each request keeps its tensors with a leading batch dimension of 1, so this
    is a concat rather than a stack. Shapes must already agree; a mismatch means
    the batch-compatibility key let through something it should have split.
    """
    head = tensors[0]
    for index, tensor in enumerate(tensors[1:], start=1):
        if tensor.shape[1:] != head.shape[1:]:
            raise ValueError(
                f"Incompatible {name} shapes in one batch: request 0 has {tuple(head.shape)}, "
                f"request {index} has {tuple(tensor.shape)}."
            )
    if len(tensors) == 1:
        return head
    return torch.cat(tensors, dim=0)


@dataclass(slots=True)
class StepInputBatch:
    """Contiguous view over the requests running one denoise step together."""

    states: list[RunnerState]
    latents: torch.Tensor
    condition: torch.Tensor
    first_frame_mask: torch.Tensor | None
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    timesteps: torch.Tensor
    guidance_scale: float

    @property
    def num_reqs(self) -> int:
        return len(self.states)

    @property
    def req_ids(self) -> list[str]:
        return [state.req_id for state in self.states]

    @classmethod
    def make_batch(cls, states: Sequence[RunnerState]) -> "StepInputBatch":
        if not states:
            raise ValueError("StepInputBatch requires at least one request.")

        latents = _stack(_require(states, "latents"), "latents")
        condition = _stack([state.extra["condition"] for state in states], "condition")
        prompt_embeds = _stack(_require(states, "prompt_embeds"), "prompt_embeds")

        masks = [state.extra.get("first_frame_mask") for state in states]
        if any(mask is None for mask in masks) and any(mask is not None for mask in masks):
            raise ValueError("Cannot batch requests that disagree on whether a first-frame mask is used.")
        first_frame_mask = None if masks[0] is None else _stack(masks, "first_frame_mask")

        negatives = [state.negative_prompt_embeds for state in states]
        if any(item is None for item in negatives) and any(item is not None for item in negatives):
            raise ValueError("Cannot batch requests that disagree on classifier-free guidance.")
        negative_prompt_embeds = None if negatives[0] is None else _stack(negatives, "negative_prompt_embeds")

        # Per-request current timestep. Requests at different progress contribute
        # different rows, which is what lets a request join a running batch.
        timesteps = torch.stack([cls._current_timestep(state) for state in states]).to(latents.device)

        guidance_scales = {float(state.extra["guidance_scale"]) for state in states}
        if len(guidance_scales) != 1:
            raise ValueError(f"Cannot batch requests with differing guidance scales: {sorted(guidance_scales)}.")

        return cls(
            states=list(states),
            latents=latents,
            condition=condition,
            first_frame_mask=first_frame_mask,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            timesteps=timesteps,
            guidance_scale=guidance_scales.pop(),
        )

    @staticmethod
    def _current_timestep(state: RunnerState) -> torch.Tensor:
        timestep = state.current_timestep
        if timestep is None:
            raise ValueError(f"Request {state.req_id} has no timestep left to run.")
        return timestep

    def slice_for(self, index: int, tensor: torch.Tensor) -> torch.Tensor:
        """Take the rows of a batched tensor belonging to one request."""
        rows = self.states[index].latents.shape[0]
        start = sum(state.latents.shape[0] for state in self.states[:index])
        return tensor[start : start + rows]
