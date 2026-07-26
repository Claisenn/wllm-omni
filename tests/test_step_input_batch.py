"""Tests for denoise-step batch assembly and the batched executor path."""

from __future__ import annotations

import torch

from wllm_omni.models.diffusion_executor import DiffusionExecutor
from wllm_omni.worker.input_batch import StepInputBatch
from wllm_omni.worker.utils import RunnerState

from tests.test_step_batch_compat import make_request

LATENT_SHAPE = (1, 4, 3, 8, 8)
EMBED_SHAPE = (1, 6, 16)


def make_runner_state(req_id: str, *, step_index: int = 0, num_steps: int = 4, guidance_scale: float = 5.0):
    request = make_request(num_inference_steps=num_steps, guidance_scale=guidance_scale)
    state = RunnerState(
        req_id=req_id,
        sampling=request.sampling_params,
        prompt=request.prompt,
        image=request.image,
        negative_prompt=request.sampling_params.negative_prompt,
    )
    state.latents = torch.randn(LATENT_SHAPE)
    state.prompt_embeds = torch.randn(EMBED_SHAPE)
    state.negative_prompt_embeds = torch.randn(EMBED_SHAPE)
    state.timesteps = torch.linspace(1000, 0, num_steps)
    state.step_index = step_index
    state.extra["condition"] = torch.randn(LATENT_SHAPE)
    state.extra["first_frame_mask"] = torch.ones(LATENT_SHAPE)
    state.extra["guidance_scale"] = guidance_scale
    return state


class TestStepInputBatch:
    def test_gathers_rows_from_every_request(self):
        states = [make_runner_state(f"req-{i}") for i in range(3)]

        batch = StepInputBatch.make_batch(states)

        assert batch.num_reqs == 3
        assert batch.req_ids == ["req-0", "req-1", "req-2"]
        assert batch.latents.shape == (3, *LATENT_SHAPE[1:])
        assert batch.condition.shape == (3, *LATENT_SHAPE[1:])
        assert batch.prompt_embeds.shape == (3, *EMBED_SHAPE[1:])
        assert batch.negative_prompt_embeds.shape == (3, *EMBED_SHAPE[1:])
        assert batch.first_frame_mask.shape == (3, *LATENT_SHAPE[1:])

    def test_single_request_batch_is_a_passthrough(self):
        """N=1 must not concatenate, so the default path stays bit-identical."""
        state = make_runner_state("req-0")

        batch = StepInputBatch.make_batch([state])

        assert batch.latents is state.latents
        assert batch.prompt_embeds is state.prompt_embeds

    def test_requests_at_different_progress_contribute_their_own_timestep(self):
        """This is what allows a request to join a batch already in flight."""
        states = [
            make_runner_state("req-0", step_index=0),
            make_runner_state("req-1", step_index=2),
        ]

        batch = StepInputBatch.make_batch(states)

        assert batch.timesteps.shape == (2,)
        assert torch.equal(batch.timesteps[0], states[0].timesteps[0])
        assert torch.equal(batch.timesteps[1], states[1].timesteps[2])

    def test_slice_for_recovers_each_request_rows(self):
        states = [make_runner_state(f"req-{i}") for i in range(3)]
        batch = StepInputBatch.make_batch(states)

        for index, state in enumerate(states):
            assert torch.equal(batch.slice_for(index, batch.latents), state.latents)

    def test_mismatched_shapes_are_rejected(self):
        states = [make_runner_state("req-0"), make_runner_state("req-1")]
        states[1].latents = torch.randn(1, 4, 3, 16, 16)

        try:
            StepInputBatch.make_batch(states)
        except ValueError as exc:
            assert "Incompatible latents shapes" in str(exc)
        else:
            raise AssertionError("expected a shape mismatch to be rejected")

    def test_unprepared_request_is_rejected(self):
        state = make_runner_state("req-0")
        state.latents = None

        try:
            StepInputBatch.make_batch([state])
        except ValueError as exc:
            assert "prepare_encode" in str(exc)
        else:
            raise AssertionError("expected an unprepared request to be rejected")


class RecordingPipeline:
    """Fake Wan pipeline recording how the executor drives it."""

    def __init__(self):
        self.config = type("Config", (), {"enable_profiling": False, "use_cpu_offload": False})()
        self.denoise_batch_sizes: list[int] = []
        self.prepared: list[str] = []
        self.decoded: list[str] = []

    def prepare_encode(self, state: RunnerState) -> RunnerState:
        self.prepared.append(state.req_id)
        num_steps = state.sampling.num_inference_steps
        state.latents = torch.zeros(LATENT_SHAPE)
        state.prompt_embeds = torch.zeros(EMBED_SHAPE)
        state.negative_prompt_embeds = torch.zeros(EMBED_SHAPE)
        state.timesteps = torch.linspace(1000, 0, num_steps)
        state.step_index = 0
        state.extra["condition"] = torch.zeros(LATENT_SHAPE)
        state.extra["first_frame_mask"] = torch.ones(LATENT_SHAPE)
        state.extra["guidance_scale"] = state.sampling.guidance_scale
        return state

    def denoise_step(self, batch: StepInputBatch) -> torch.Tensor:
        self.denoise_batch_sizes.append(batch.num_reqs)
        return torch.ones_like(batch.latents)

    def step_scheduler(self, batch: StepInputBatch, noise_pred: torch.Tensor) -> None:
        for index, state in enumerate(batch.states):
            state.latents = state.latents + batch.slice_for(index, noise_pred)
            state.step_index += 1

    def post_decode(self, state: RunnerState):
        self.decoded.append(state.req_id)
        return f"video-{state.req_id}"


class TestBatchedExecutor:
    @staticmethod
    def _init(executor, name: str, num_steps: int):
        """Init a state whose request id and scheduler id are both ``name``."""
        request = make_request(num_inference_steps=num_steps)
        request.request_id = name
        return executor.init_state(name, request)

    @staticmethod
    def _drive(executor, states, rounds):
        results = []
        active = list(states)
        for _ in range(rounds):
            if not active:
                break
            forward_batch = executor.build_forward_batch(active)
            output = executor.forward(forward_batch)
            executor.update_states(active, output)
            results.append(output.outputs)
            active = [s for s, item in zip(active, output.outputs, strict=True) if not item.finished]
        return results

    def test_one_denoise_call_serves_the_whole_batch(self):
        pipeline = RecordingPipeline()
        executor = DiffusionExecutor(pipeline)
        states = [self._init(executor, f"sched-{i}", 3) for i in range(3)]

        self._drive(executor, states, rounds=3)

        # Three denoise rounds, each a single call covering all three requests.
        assert pipeline.denoise_batch_sizes == [3, 3, 3]
        # prepare_encode stays per-request: it is per-request work hitting
        # per-request caches, and it runs exactly once each.
        assert pipeline.prepared == ["sched-0", "sched-1", "sched-2"]

    def test_requests_finish_independently(self):
        pipeline = RecordingPipeline()
        executor = DiffusionExecutor(pipeline)
        short = self._init(executor, "sched-short", 2)
        long = self._init(executor, "sched-long", 4)

        rounds = self._drive(executor, [short, long], rounds=4)

        finished_at = {}
        for round_index, outputs in enumerate(rounds):
            for item in outputs:
                if item.finished and item.req_id not in finished_at:
                    finished_at[item.req_id] = round_index
        assert finished_at["sched-short"] < finished_at["sched-long"]
        assert set(pipeline.decoded) == {"sched-short", "sched-long"}

    def test_batch_shrinks_as_requests_complete(self):
        pipeline = RecordingPipeline()
        executor = DiffusionExecutor(pipeline)
        states = [
            self._init(executor, "sched-short", 2),
            self._init(executor, "sched-long", 4),
        ]

        self._drive(executor, states, rounds=4)

        # Both requests denoise together until the short one completes, after
        # which the long one continues alone.
        assert pipeline.denoise_batch_sizes == [2, 2, 1, 1]

    def test_results_are_returned_per_request(self):
        pipeline = RecordingPipeline()
        executor = DiffusionExecutor(pipeline)
        states = [self._init(executor, f"sched-{i}", 1) for i in range(2)]

        rounds = self._drive(executor, states, rounds=1)

        results = {item.req_id: item.result for item in rounds[0]}
        assert results == {"sched-0": "video-sched-0", "sched-1": "video-sched-1"}
