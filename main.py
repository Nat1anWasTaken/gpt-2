from pathlib import Path

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from torch import nn

MODEL_DIR = Path(__file__).resolve().parent


def load_weights() -> dict[str, torch.Tensor]:
    weights = load_file(MODEL_DIR / "model.safetensors")

    return weights


def load_tokenizer() -> Tokenizer:
    return Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))


def load_first_layer_norm(
    weights: dict[str, torch.Tensor], layer_index: int
) -> nn.LayerNorm:
    layer = nn.LayerNorm(768)

    with torch.no_grad():
        layer.weight.copy_(weights[f"h.{layer_index}.ln_1.weight"])
        layer.bias.copy_(weights[f"h.{layer_index}.ln_1.bias"])

    return layer


def load_second_layer_norm(
    weights: dict[str, torch.Tensor], layer_index: int
) -> nn.LayerNorm:
    layer = nn.LayerNorm(768)

    with torch.no_grad():
        layer.weight.copy_(weights[f"h.{layer_index}.ln_2.weight"])
        layer.bias.copy_(weights[f"h.{layer_index}.ln_2.bias"])

    return layer


def load_final_layer_norm(weights: dict[str, torch.Tensor]) -> nn.LayerNorm:
    layer = nn.LayerNorm(768)

    with torch.no_grad():
        layer.weight.copy_(weights["ln_f.weight"])
        layer.bias.copy_(weights["ln_f.bias"])

    return layer


def load_embedding_layers(
    weights: dict[str, torch.Tensor],
) -> tuple[nn.Embedding, nn.Embedding]:
    """Build token and position embedding layers from pretrained weights."""
    word_token_embedding = nn.Embedding(50257, 768)
    word_position_embedding = nn.Embedding(1024, 768)

    with torch.no_grad():
        word_token_embedding.weight.copy_(weights["wte.weight"])
        word_position_embedding.weight.copy_(weights["wpe.weight"])

    return word_token_embedding, word_position_embedding


def embed_text(
    text: str,
    tokenizer: Tokenizer,
    wte: nn.Embedding,
    wpe: nn.Embedding,
) -> torch.Tensor:
    """Tokenize text and return its pretrained GPT-2 token plus position embeddings."""
    token_ids = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)

    if token_ids.numel() > 1024:
        raise ValueError("Text exceeds GPT-2's 1024-token context length")

    positions = torch.arange(token_ids.numel())  # Assign positions 0, 1, 2, and so on.
    return wte(token_ids) + wpe(positions)


def mlp_first_projection(
    normalized: torch.Tensor, weights: dict[str, torch.Tensor], layer_index: int
) -> torch.Tensor:
    """Expand each token from 768 to 3072 features."""
    return (
        torch.matmul(normalized, weights[f"h.{layer_index}.mlp.c_fc.weight"])
        + weights[f"h.{layer_index}.mlp.c_fc.bias"]
    )


def mlp_second_projection(
    activated: torch.Tensor, weights: dict[str, torch.Tensor], layer_index: int
) -> torch.Tensor:
    """Project the MLP output from 3072 back to 768 features."""
    return (
        torch.matmul(activated, weights[f"h.{layer_index}.mlp.c_proj.weight"])
        + weights[f"h.{layer_index}.mlp.c_proj.bias"]
    )


def transformer_block(
    hidden: torch.Tensor, weights: dict[str, torch.Tensor], layer_index: int
) -> torch.Tensor:
    """Apply one GPT-2 transformer block using its own pretrained weights."""
    prefix = f"h.{layer_index}"
    normalized = load_first_layer_norm(weights, layer_index)(hidden)

    qkv = (
        torch.matmul(normalized, weights[f"{prefix}.attn.c_attn.weight"])
        + weights[f"{prefix}.attn.c_attn.bias"]
    )
    q, k, v = qkv.chunk(3, dim=-1)

    query = q.reshape(q.shape[0], 12, 64).transpose(0, 1)
    key = k.reshape(k.shape[0], 12, 64).transpose(0, 1)
    value = v.reshape(v.shape[0], 12, 64).transpose(0, 1)

    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / 8
    token_count = attention_scores.shape[-1]
    causal_mask = torch.ones(
        token_count, token_count, dtype=torch.bool, device=attention_scores.device
    ).tril()
    attention_weights = torch.softmax(
        attention_scores.masked_fill(~causal_mask, float("-inf")), dim=-1
    )
    context = torch.matmul(attention_weights, value)
    combined_context = context.transpose(0, 1).reshape(token_count, 768)

    projected_attention = (
        torch.matmul(combined_context, weights[f"{prefix}.attn.c_proj.weight"])
        + weights[f"{prefix}.attn.c_proj.bias"]
    )
    after_attention = hidden + projected_attention

    normalized_for_mlp = load_second_layer_norm(weights, layer_index)(after_attention)
    mlp_expanded = mlp_first_projection(normalized_for_mlp, weights, layer_index)
    mlp_activated = torch.nn.functional.gelu(mlp_expanded, approximate="tanh")
    mlp_projected = mlp_second_projection(mlp_activated, weights, layer_index)
    return after_attention + mlp_projected


def project_to_vocabulary(hidden: torch.Tensor, wte: nn.Embedding) -> torch.Tensor:
    """Use the token embedding weights to produce one logit per vocabulary token."""
    return torch.matmul(hidden, wte.weight.T)


def gpt2_complete(
    input: list[str],
    max_seq_length: int = 1024,
) -> tuple[list[str], torch.Tensor]:
    """Generate greedy completions with a from-scratch GPT-2 Small implementation.

    Loads the pretrained GPT-2 Small weights into a manually implemented
    transformer consisting of token and positional embeddings, causal
    multi-head self-attention, feed-forward layers, residual connections,
    layer normalization, and a language-modeling output head.

    Generation is performed for the entire batch simultaneously. At each
    decoding step, the highest-logit token is selected for every unfinished
    sequence. A sequence stops generating after producing the EOS token, and
    generation terminates once every sequence has either produced EOS or
    reached ``max_seq_length``.

    Args:
        input: Batch of input strings to complete.
        max_seq_length: Maximum total tokenized sequence length, including
            both prompt and generated tokens.

    Returns:
        A tuple containing:
            - The decoded completion for each input string.
            - The model logits used during greedy generation.
    """
    raise NotImplementedError


def main():
    weights = load_weights()
    tokenizer = load_tokenizer()
    wte, wpe = load_embedding_layers(weights)

    # Token Embedding + Positional Embedding
    hidden = embed_text("Hello world, are you ok", tokenizer, wte, wpe)

    # Transformer Blocks
    for layer_index in range(12):
        hidden = transformer_block(hidden, weights, layer_index)

    # Final Layer & Output
    final_ln = load_final_layer_norm(weights)
    final_hidden = final_ln(hidden)
    logits = project_to_vocabulary(final_hidden, wte)
    print(logits.shape)  # [T, 50257]

if __name__ == "__main__":
    main()
