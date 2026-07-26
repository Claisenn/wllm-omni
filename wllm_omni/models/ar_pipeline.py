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
    """Mutable per-request state for stepwise (prefill + decode) generation.

    ``cache`` and ``attention_mask`` are backend-specific (for
    TransformersARPipeline: a transformers ``Cache``/``past_key_values``
    object and the growing attention mask tensor). Executors only ever read
    ``finished``; everything else is opaque and owned by the pipeline that
    produced it.
    """

    request_id: str
    input_length: int
    generated_ids: list[int] = field(default_factory=list)
    finished: bool = False
    cache: Any = None
    attention_mask: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ARPipeline(ABC):
    """Text-generation stage interface for mini-Omni composition.

    ``generate`` is the only required method: it runs a request to
    completion in one call. Pipelines that can decode incrementally (KV
    cache) may additionally implement ``prefill``/``decode_step`` so
    ARExecutor can drive them one step at a time instead of blocking on a
    single monolithic call; see
    ``wllm_omni.models.ar_executor.AR_STEP_EXECUTION_METHODS``.
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
    """Minimal local Transformers CausalLM backend for the AR stage."""

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
        """Run a request to completion by driving the stepwise decode loop.

        Kept as the ``ARPipeline`` fallback entry point (and for direct/manual
        use); ARExecutor normally calls ``prefill``/``decode_step`` directly
        so it can interleave steps across requests instead of blocking here.
        """
        state = self.prefill(request)
        while not state.finished:
            state = self.decode_step(state)
        return self.finalize(state)

    def prefill(self, request: OmniRequest) -> ARDecodeState:
        import torch

        prompt = self._build_prompt(request.prompt)
        inputs = self._tokenize_prompt(prompt)
        input_length = int(inputs.input_ids.shape[-1])
        with torch.no_grad():
            out = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.get("attention_mask"),
                use_cache=True,
            )
        next_id = int(torch.argmax(out.logits[:, -1, :], dim=-1))
        state = ARDecodeState(
            request_id=request.request_id,
            input_length=input_length,
            generated_ids=[next_id],
            cache=out.past_key_values,
            attention_mask=inputs.get("attention_mask"),
            metadata={"prompt": request.prompt},
        )
        state.finished = self._is_stopping(state)
        return state

    def decode_step(self, state: ARDecodeState) -> ARDecodeState:
        import torch

        if state.finished:
            return state
        last_id = state.generated_ids[-1]
        next_input_ids = torch.tensor([[last_id]], device=self.device)
        attention_mask = state.attention_mask
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
        with torch.no_grad():
            out = self.model(
                input_ids=next_input_ids,
                attention_mask=attention_mask,
                past_key_values=state.cache,
                use_cache=True,
            )
        next_id = int(torch.argmax(out.logits[:, -1, :], dim=-1))
        state.generated_ids.append(next_id)
        state.cache = out.past_key_values
        state.attention_mask = attention_mask
        state.finished = self._is_stopping(state)
        return state

    def finalize(self, state: ARDecodeState) -> ARTextOutput:
        generated_ids = state.generated_ids
        if generated_ids and generated_ids[-1] in self._stop_token_ids():
            generated_ids = generated_ids[:-1]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if not text:
            text = state.metadata.get("prompt", "").strip()
        tokens = self.tokenizer.convert_ids_to_tokens(generated_ids)
        return ARTextOutput(
            request_id=state.request_id,
            text=text,
            tokens=tokens,
            token_ids=list(generated_ids),
            metadata={
                "mode": "transformers_causal_lm",
                "model": self.model_path,
                "input_tokens": state.input_length,
                "token_count": len(generated_ids),
            },
        )

    def _is_stopping(self, state: ARDecodeState) -> bool:
        if state.generated_ids[-1] in self._stop_token_ids():
            return True
        return len(state.generated_ids) >= self.max_new_tokens

    def _stop_token_ids(self) -> set[int]:
        """Every token id that should end generation.

        Mirrors what ``model.generate()`` treats as EOS: the checkpoint's
        ``generation_config.eos_token_id`` (which some chat models declare as
        a list of multiple stop tokens, e.g. a dedicated turn-end token in
        addition to the tokenizer's own EOS) as well as the tokenizer's own
        ``eos_token_id``.
        """
        ids: set[int] = set()
        config_eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        if isinstance(config_eos, int):
            ids.add(config_eos)
        elif config_eos is not None:
            ids.update(int(item) for item in config_eos)
        if self.tokenizer.eos_token_id is not None:
            ids.add(int(self.tokenizer.eos_token_id))
        return ids

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
