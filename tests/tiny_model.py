"""A tiny random-weight Qwen2 checkpoint, so the pipeline can be tested end to end with no download.

The weights are random but the architecture is the real one (GQA attention, RoPE, RMSNorm, SwiGLU),
so the stage runner, the loaders and the cluster are exercised exactly as with a real checkpoint.
"""
import os

import torch

LAYERS = 8
VOCAB = 256


def make_tiny_model(path: str) -> str:
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    if os.path.exists(os.path.join(path, "model.safetensors")):
        return path
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        # a wide init keeps greedy logits well separated, so FP32 token equality is stable
        initializer_range=0.2,
        bos_token_id=0,
        eos_token_id=1,
    )
    Qwen2ForCausalLM(cfg).save_pretrained(path, safe_serialization=True)
    tok = Tokenizer(models.WordLevel({f"t{i}": i for i in range(VOCAB)}, unk_token="t0"))
    PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="t1", unk_token="t0").save_pretrained(path)
    return path
