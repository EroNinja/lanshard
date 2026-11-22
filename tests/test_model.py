"""Numerical correctness of the stage runner against stock Hugging Face (needs the 0.5B model)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lanshard.model_utils import HeadModules, LayerLoader, Segment, StageRunner, resolve_model_dir  # noqa: E402

MODEL = os.environ.get("LANSHARD_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float16 if DEV == "cuda" else torch.float32


def _have_model():
    """Only run when the weights are already cached (CI does not download 1 GB)."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(MODEL, local_files_only=True, allow_patterns=["*.json", "*.safetensors"])
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _have_model(), reason="model not available")


@pytest.fixture(scope="module")
def parts():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    d = resolve_model_dir(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(d, dtype=DT).to(DEV).eval()
    tok = AutoTokenizer.from_pretrained(d)
    L = LayerLoader(MODEL)
    n = L.geom.num_layers
    # two stages, as two workers would hold them
    r1, r2 = StageRunner(L.cfg, DEV, DT), StageRunner(L.cfg, DEV, DT)
    r1.layers = L.load_layers(range(0, n // 2), DEV, DT)
    r2.layers = L.load_layers(range(n // 2, n), DEV, DT)
    return hf, tok, r1, r2, HeadModules(MODEL, DEV, DT), n


def _ids(tok, q):
    ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=True)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


def test_two_stage_greedy_matches_hf(parts):
    hf, tok, r1, r2, head, n = parts
    ids = _ids(tok, "Write a haiku about Wi-Fi.")
    ref = hf.generate(torch.tensor([ids], device=DEV), max_new_tokens=24, do_sample=False, repetition_penalty=1.0,
                      top_k=None, top_p=None, temperature=None)[0, len(ids):].tolist()
    P = len(ids)
    c = P // 2  # chunked prefill
    for start, chunk in ((0, ids[:c]), (c, ids[c:])):
        x = head.embed_tokens(torch.tensor(chunk))
        seg = [Segment(7, start, len(chunk), P + 32)]
        x = r2.forward(r1.forward(x, seg, range(n // 2)), seg, range(n // 2, n))
    out = [int(head.logits(x[-1:]).argmax())]
    while len(out) < len(ref):
        seg = [Segment(7, P + len(out) - 1, 1, P + 32)]
        x = head.embed_tokens(torch.tensor([out[-1]]))
        x = r2.forward(r1.forward(x, seg, range(n // 2)), seg, range(n // 2, n))
        out.append(int(head.logits(x).argmax()))
    assert out == ref


def test_packed_batch_equals_separate(parts):
    """A packed micro-batch (prefill + decode segments) gives the same hidden states as separate runs."""
    _, tok, r1, _, head, n = parts
    a, b = _ids(tok, "Who wrote Hamlet?"), _ids(tok, "What is 2+2?")
    layers = range(n // 2)
    sep_a = r1.forward(head.embed_tokens(torch.tensor(a)), [Segment(100, 0, len(a), 64)], layers)
    sep_b = r1.forward(head.embed_tokens(torch.tensor(b)), [Segment(101, 0, len(b), 64)], layers)
    x = head.embed_tokens(torch.tensor(a + b))
    packed = r1.forward(x, [Segment(200, 0, len(a), 64), Segment(201, 0, len(b), 64)], layers)
    tol = 2e-2 if DT == torch.float16 else 1e-4
    assert torch.allclose(packed[: len(a)].float(), sep_a.float(), atol=tol, rtol=tol)
    assert torch.allclose(packed[len(a):].float(), sep_b.float(), atol=tol, rtol=tol)
