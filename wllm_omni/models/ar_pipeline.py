from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from wllm_omni.request import OmniRequest


@dataclass(slots=True)
class ARTextOutput:
    request_id: str
    text: str
    tokens: list[str]
    token_ids: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ARDecodeState:
    """Per-request state for stepwise (prefill + KV-cache decode) generation.

    Produced by ``ARPipeline.prefill`` and threaded through ``decode_step``
    until ``finished`` is True, then turned into an ``ARTextOutput`` by
    ``finalize``. Executors only ever read ``finished``; every other field is
    owned by the pipeline that produced the state. For
    ``TransformersARPipeline`` the ``Any`` fields hold torch tensors, a
    transformers KV ``Cache``, and the ``LogitsProcessorList`` /
    ``StoppingCriteriaList`` built by transformers' own generation machinery
    at prefill time (built once, reused every step).
    """

    request_id: str
    prompt: str
    input_length: int
    input_ids: Any
    attention_mask: Any
    logits_processor: Any
    stopping_criteria: Any
    cache: Any = None
    finished: bool = False


class ARPipeline(ABC):
    """Text-generation stage interface for mini-Omni composition.

    ``generate`` is the only required method: run one request to completion.
    Pipelines that can decode incrementally may additionally implement the
    stepwise contract -- ``prefill`` / ``decode_step`` / ``finalize`` (see
    ``wllm_omni.models.ar_executor.AR_STEP_EXECUTION_METHODS``) -- so that
    ARExecutor can advance one decode step per ``forward()`` call instead of
    blocking on a single monolithic call.
    """

    @abstractmethod
    def generate(self, request: OmniRequest) -> ARTextOutput:
        pass


class IdentityARPipeline(ARPipeline):
    """Deterministic AR placeholder used before a real causal LM backend."""

    def generate(self, request: OmniRequest) -> ARTextOutput:
        text = self._normalize_prompt(request.prompt)
        tokens = self._tokenize(text)
        return ARTextOutput(
            request_id=request.request_id,
            text=text,
            tokens=tokens,
            token_ids=[self._stable_token_id(token) for token in tokens],
            metadata={
                "mode": "identity_prompt_bridge",
                "token_count": len(tokens),
            },
        )

    @staticmethod
    def _normalize_prompt(prompt: str) -> str:
        text = " ".join(prompt.strip().split())
        return text or "high quality video"

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.split()

    @staticmethod
    def _stable_token_id(token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)


class TransformersARPipeline(ARPipeline):
    """Local Transformers CausalLM backend with explicit prefill/decode split.

    The stepwise loop must be bit-identical to
    ``model.generate(do_sample=False, max_new_tokens=..., pad_token_id=...)``.
    To guarantee that, ``prefill`` builds the logits processors and stopping
    criteria through transformers' own ``_prepare_generation_config`` /
    ``_get_logits_processor`` / ``_get_stopping_criteria`` -- so checkpoint
    generation_config settings such as ``repetition_penalty``,
    ``no_repeat_ngram_size``, ``min_new_tokens``, and multi-token
    ``eos_token_id`` lists behave exactly as they do under ``generate()``
    instead of being re-implemented (and drifting) here.
    """

    def __init__(
        self,
        model: str,
        *,
        device: str = "cuda",
        dtype: Any = None,
        local_files_only: bool = True,
        max_new_tokens: int = 64,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = model
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.dtype = dtype or torch.bfloat16
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=local_files_only, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            dtype=self.dtype,
            local_files_only=local_files_only,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def generate(self, request: OmniRequest) -> ARTextOutput:
        """Run one request to completion via the stepwise primitives.

        Kept as the required ``ARPipeline`` entry point for direct use;
        ARExecutor calls ``prefill``/``decode_step``/``finalize`` itself so it
        can advance one step per ``forward()`` call.
        """
        state = self.prefill(request)
        while not state.finished:
            state = self.decode_step(state)
        return self.finalize(state)

    def prefill(self, request: OmniRequest) -> ARDecodeState:
        """Tokenize, run the full-prompt forward, and emit the first token."""
        import torch
        from transformers.generation import LogitsProcessorList, StoppingCriteriaList

        prompt = self._build_prompt(request.prompt)
        inputs = self._tokenize_prompt(prompt)
        input_ids = inputs.input_ids
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        input_length = int(input_ids.shape[-1])

        # Build the exact processor/criteria stack model.generate() would use
        # for the equivalent call. _prepare_generation_config also validates
        # arguments (e.g. rejects max_new_tokens<=0) exactly like generate().
        generation_config, _ = self.model._prepare_generation_config(
            None,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generation_config.max_length = input_length + self.max_new_tokens
        self.model._prepare_special_tokens(
            generation_config,
            kwargs_has_attention_mask=True,
            device=input_ids.device,
            batch_size=int(input_ids.shape[0]),
        )
        logits_processor = self.model._get_logits_processor(
            generation_config,
            input_ids_seq_length=input_length,
            encoder_input_ids=None,
            prefix_allowed_tokens_fn=None,
            logits_processor=LogitsProcessorList(),
            device=input_ids.device,
        )
        stopping_criteria = self.model._get_stopping_criteria(generation_config, StoppingCriteriaList())

        state = ARDecodeState(
            request_id=request.request_id,
            prompt=request.prompt,
            input_length=input_length,
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
        )
        return self._step(state)

    def decode_step(self, state: ARDecodeState) -> ARDecodeState:
        """Advance one token using the KV cache built by prefill."""
        if state.finished:
            return state
        return self._step(state)

    def finalize(self, state: ARDecodeState) -> ARTextOutput:
        """Decode the generated ids into the stage output.

        Matches the old ``model.generate()``-based path exactly: ``token_ids``
        keep a trailing stop token if one was generated (generate() includes
        it in the returned sequence); only ``text`` drops it, via
        ``skip_special_tokens``.
        """
        generated_ids = state.input_ids[0, state.input_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if not text:
            text = state.prompt.strip()
        token_ids = [int(item) for item in generated_ids.detach().cpu().tolist()]
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        return ARTextOutput(
            request_id=state.request_id,
            text=text,
            tokens=tokens,
            token_ids=token_ids,
            metadata={
                "mode": "transformers_causal_lm",
                "model": self.model_path,
                "input_tokens": state.input_length,
                "token_count": len(token_ids),
            },
        )

    def _step(self, state: ARDecodeState) -> ARDecodeState:
        """One model forward + logits processors + argmax + stop check.

        Mirrors one iteration of transformers' greedy ``_sample`` loop: the
        last-position logits are copied to float32 before the processors run
        (so lower-precision checkpoints tie-break identically), the processors
        see the full sequence so far, and the stopping criteria run after the
        token is appended -- which is why generate() includes a generated EOS
        in its output.
        """
        import torch

        step_input_ids = state.input_ids if state.cache is None else state.input_ids[:, -1:]
        with torch.no_grad():
            out = self.model(
                input_ids=step_input_ids,
                attention_mask=state.attention_mask,
                past_key_values=state.cache,
                use_cache=True,
            )
        state.cache = out.past_key_values
        logits = out.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=state.input_ids.device)
        scores = state.logits_processor(state.input_ids, logits)
        next_token = torch.argmax(scores, dim=-1)
        state.input_ids = torch.cat([state.input_ids, next_token[:, None]], dim=-1)
        state.attention_mask = torch.cat(
            [state.attention_mask, state.attention_mask.new_ones((state.attention_mask.shape[0], 1))],
            dim=-1,
        )
        is_done = state.stopping_criteria(state.input_ids, scores)
        state.finished = bool(torch.as_tensor(is_done).all())
        return state

    def _tokenize_prompt(self, prompt: str):
        messages = [
            {"role": "system", "content": "You rewrite user requests into concise visual prompts for image-to-video generation."},
            {"role": "user", "content": prompt},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.device)
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)

    @staticmethod
    def _build_prompt(prompt: str) -> str:
        return (
            "Rewrite the following image-to-video request as a concise, visual video generation prompt. "
            "Keep the main subject, scene, motion, and style. Return only the rewritten prompt.\n\n"
            f"Request: {prompt.strip()}\nPrompt:"
        )
