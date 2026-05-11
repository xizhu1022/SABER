"""Subject-model registry.

One logical id per backbone; every script/loader references the logical id
rather than the raw Hugging Face repo. Switching the weight source is a
one-line edit here.

Fields:
  hf_repo          Hugging Face repo id passed to ``AutoModelForCausalLM.from_pretrained``.
  dtype            torch dtype string consumed by ``getattr(torch, dtype)``.
  n_layers         Number of transformer layers.
  hidden_size      Probe input dim fallback; scripts prefer ``model.config.hidden_size``.
  layer_fractions  Relative layer positions in [0, 1] used to resolve absolute indices.
                   0.0 = embeddings / layer 0 output, 1.0 = last layer output.
                   Absolute index = round(f * n_layers).
  token_positions  Where hidden states are read at each forward pass.
  chat_template    ``"auto"`` uses ``tokenizer.apply_chat_template``.
  notes            Free-form metadata.

The alias key is what every downstream script refers to (e.g., ``"llama-3.1-8b-instruct"``).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    hf_repo: str
    dtype: str = "float16"
    n_layers: int = 32
    hidden_size: int = 4096
    layer_fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    probe_layers_single: tuple[float, ...] = (0.25, 0.5, 0.75)
    token_positions: tuple[str, ...] = (
        "last_q",        # last token of the user question
        "mean_q",        # mean over question tokens
        "first_a",       # first generated answer token (post-generation hidden state)
        "last_prompt",   # last token of the prompt (the "ready to generate" state)
    )
    chat_template: str = "auto"
    notes: str = ""

    def resolve_layer(self, fraction: float) -> int:
        """Map a relative fraction in [0, 1] to an absolute hidden-states index.

        ``output_hidden_states=True`` returns ``n_layers + 1`` tensors (layer 0 =
        embeddings, layer n = final block). ``fraction=0.0`` -> embeddings;
        ``fraction=1.0`` -> last block output.
        """
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"layer fraction must be in [0,1], got {fraction}")
        return round(fraction * self.n_layers)


MODELS: dict[str, ModelSpec] = {
    "llama-3.1-8b-instruct": ModelSpec(
        hf_repo="meta-llama/Llama-3.1-8B-Instruct",
        dtype="float16",
        n_layers=32,
        hidden_size=4096,
        notes="One of the four backbones evaluated in the paper.",
    ),
    "llama-3.2-3b-instruct": ModelSpec(
        hf_repo="meta-llama/Llama-3.2-3B-Instruct",
        dtype="bfloat16",
        n_layers=28,
        hidden_size=3072,
        notes="Smaller Llama-3 family member; same tokenizer as Llama-3.1-8B-Instruct.",
    ),
    "qwen2.5-7b-instruct": ModelSpec(
        hf_repo="Qwen/Qwen2.5-7B-Instruct",
        dtype="bfloat16",
        n_layers=28,
        hidden_size=3584,
        notes="Mid-size Qwen 2.5 family member.",
    ),
    "qwen2.5-3b-instruct": ModelSpec(
        hf_repo="Qwen/Qwen2.5-3B-Instruct",
        dtype="bfloat16",
        n_layers=36,
        hidden_size=2048,
        notes="Smaller Qwen 2.5 family member.",
    ),
}

DEFAULT_MODEL_KEY = "llama-3.1-8b-instruct"


def get_model_spec(key: str | None = None) -> ModelSpec:
    spec = MODELS.get(key or DEFAULT_MODEL_KEY)
    if spec is None:
        raise KeyError(
            f"Unknown model key {key!r}. Known keys: {sorted(MODELS)}"
        )
    return spec
