"""Loading slices of a Hugging Face decoder-only model and running them.

A worker never materialises the whole model: it builds the architecture on the
``meta`` device (no memory), pulls out the decoder layers it owns, and fills only
those from the safetensors shards. The coordinator does the same for the
embedding, final norm and LM head.

The forward pass reuses the Hugging Face sub-modules (norms, projections, MLP,
rotary embedding) so the weights and numerics are the stock ones, but attention
is our own: it runs over a *packed* batch made of several requests' segments
(prefill chunks and single decode tokens mixed), each with its own KV cache and
positions. That is what lets one pipeline message carry a continuous batch.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F

_DTYPES = {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16,
           "bf16": torch.bfloat16, "float32": torch.float32, "fp32": torch.float32}


def parse_dtype(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    return _DTYPES[name.lower()]


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_dtype(device: str) -> torch.dtype:
    # FP16 matmuls on CPU are slow or unsupported; compute in FP32 there and cast
    # to the wire dtype at the stage boundary.
    return torch.float32 if device.startswith("cpu") else torch.float16


_resolve_lock = threading.Lock()


def resolve_model_dir(model_id: str) -> str:
    """Local directory holding config + safetensors (downloads if needed)."""
    if os.path.isdir(model_id):
        return model_id
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.model"]
    with _resolve_lock:
        try:
            return snapshot_download(model_id, allow_patterns=patterns, local_files_only=True)
        except Exception:
            return snapshot_download(model_id, allow_patterns=patterns)


def load_config(model_id: str):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(resolve_model_dir(model_id))


@dataclass
class ModelGeometry:
    num_layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    tied: bool

    @classmethod
    def from_config(cls, cfg) -> "ModelGeometry":
        heads = cfg.num_attention_heads
        return cls(
            num_layers=cfg.num_hidden_layers,
            hidden=cfg.hidden_size,
            heads=heads,
            kv_heads=getattr(cfg, "num_key_value_heads", heads) or heads,
            head_dim=getattr(cfg, "head_dim", None) or cfg.hidden_size // heads,
            vocab=cfg.vocab_size,
            tied=bool(getattr(cfg, "tie_word_embeddings", False)),
        )

    def kv_bytes_per_token_per_layer(self, dtype: torch.dtype) -> int:
        return 2 * self.kv_heads * self.head_dim * torch.tensor([], dtype=dtype).element_size()


class WeightIndex:
    """Maps tensor names to safetensors files and reads selected tensors."""

    def __init__(self, model_dir: str):
        self.dir = model_dir
        idx = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx, "r", encoding="utf-8") as f:
                self.weight_map: Dict[str, str] = json.load(f)["weight_map"]
        else:
            from safetensors import safe_open

            single = os.path.join(model_dir, "model.safetensors")
            with safe_open(single, framework="pt") as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}

    def keys_with_prefix(self, prefix: str) -> List[str]:
        return [k for k in self.weight_map if k.startswith(prefix)]

    def read(self, keys: Iterable[str]) -> Dict[str, torch.Tensor]:
        from safetensors import safe_open

        by_file: Dict[str, List[str]] = {}
        for k in keys:
            by_file.setdefault(self.weight_map[k], []).append(k)
        out: Dict[str, torch.Tensor] = {}
        for fname, ks in by_file.items():
            with safe_open(os.path.join(self.dir, fname), framework="pt") as f:
                for k in ks:
                    out[k] = f.get_tensor(k)
        return out


def _meta_model(cfg, dtype):
    from transformers import AutoModelForCausalLM

    with torch.device("meta"):
        try:
            return AutoModelForCausalLM.from_config(cfg, dtype=dtype)
        except TypeError:  # older transformers
            return AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype)


def _materialise(module: torch.nn.Module, state: Dict[str, torch.Tensor], device, dtype):
    module.to_empty(device=device)
    state = {k: v.to(device=device, dtype=dtype) for k, v in state.items()}
    missing, unexpected = module.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected weights: {unexpected[:5]}")
    # parameters/buffers not in the checkpoint must not be silently left as garbage
    real_missing = [m for m in missing if not m.endswith("inv_freq")]
    if real_missing:
        raise RuntimeError(f"missing weights: {real_missing[:5]}")
    module.eval()
    module.requires_grad_(False)
    return module


class LayerLoader:
    """Builds individual decoder layers of ``model_id`` with real weights."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.dir = resolve_model_dir(model_id)
        self.cfg = load_config(model_id)
        self.geom = ModelGeometry.from_config(self.cfg)
        self.index = WeightIndex(self.dir)
        self._prefix = self._detect_layer_prefix()

    def _detect_layer_prefix(self) -> str:
        for cand in ("model.layers.", "model.language_model.layers.", "transformer.h."):
            if any(k.startswith(cand) for k in self.index.weight_map):
                return cand
        raise RuntimeError("unrecognised checkpoint layout")

    def load_layers(self, layer_ids: Sequence[int], device: str, dtype: torch.dtype) -> Dict[int, torch.nn.Module]:
        meta = _meta_model(self.cfg, dtype)
        layers = meta.model.layers
        out = {}
        for i in layer_ids:
            prefix = f"{self._prefix}{i}."
            raw = self.index.read(self.index.keys_with_prefix(prefix))
            state = {k[len(prefix):]: v for k, v in raw.items()}
            out[i] = _materialise(layers[i], state, device, dtype)
        del meta
        return out

    def layer_param_bytes(self, layer_id: int = 0, dtype: torch.dtype = torch.float16) -> int:
        """Bytes one layer occupies at ``dtype`` (computed from the checkpoint shapes)."""
        from safetensors import safe_open

        prefix = f"{self._prefix}{layer_id}."
        total = 0
        el = torch.tensor([], dtype=dtype).element_size()
        by_file: Dict[str, List[str]] = {}
        for k in self.index.keys_with_prefix(prefix):
            by_file.setdefault(self.index.weight_map[k], []).append(k)
        for fname, ks in by_file.items():
            with safe_open(os.path.join(self.dir, fname), framework="pt") as f:
                for k in ks:
                    n = 1
                    for s in f.get_slice(k).get_shape():
                        n *= s
                    total += n * el
        return total


class HeadModules:
    """Embedding, final norm, LM head and rotary embedding, owned by the coordinator."""

    def __init__(self, model_id: str, device: str, dtype: torch.dtype):
        loader = LayerLoader(model_id)
        self.cfg, self.geom = loader.cfg, loader.geom
        meta = _meta_model(self.cfg, dtype)
        idx = loader.index
        emb_key = next(k for k in idx.weight_map if k.endswith("embed_tokens.weight"))
        norm_key = emb_key.replace("embed_tokens.weight", "norm.weight")
        keys = [emb_key, norm_key]
        has_head = "lm_head.weight" in idx.weight_map
        if has_head:
            keys.append("lm_head.weight")
        t = idx.read(keys)
        self.embed = _materialise(meta.model.embed_tokens, {"weight": t[emb_key]}, device, dtype)
        self.norm = _materialise(meta.model.norm, {"weight": t[norm_key]}, device, dtype)
        head_w = t["lm_head.weight"] if has_head else t[emb_key]
        self.lm_head_weight = head_w.to(device=device, dtype=dtype)
        self.device, self.dtype = device, dtype
        del meta

    @torch.no_grad()
    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embed(ids.to(self.device))

    @torch.no_grad()
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        h = self.norm(hidden.to(self.device, self.dtype))
        return F.linear(h, self.lm_head_weight).float()


def make_rotary(cfg, device: str):
    """Instantiate the architecture's own rotary embedding (handles llama3/yarn scaling)."""
    meta = _meta_model(cfg, torch.float32)
    cls = type(meta.model.rotary_emb)
    del meta
    try:
        return cls(config=cfg, device=device)
    except TypeError:  # transformers >= 5 dropped the device argument
        return cls(config=cfg).to(device)


try:  # torch >= 2.5 does grouped-query attention natively (no K/V copy)
    F.scaled_dot_product_attention(torch.zeros(1, 2, 1, 4), torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4),
                                   enable_gqa=True)
    _NATIVE_GQA = True
except TypeError:
    _NATIVE_GQA = False


def _sdpa(q, k, v, gqa: bool, **kw):
    if gqa:
        if _NATIVE_GQA:
            return F.scaled_dot_product_attention(q, k, v, enable_gqa=True, **kw)
        rep = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    return F.scaled_dot_product_attention(q, k, v, **kw)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


@dataclass
class Segment:
    """A contiguous run of tokens from one request inside a packed micro-batch."""

    rid: int
    start: int      # absolute position of the first token in this segment
    n: int          # number of tokens
    max_len: int    # KV capacity to reserve when the request is first seen

    def to_list(self):
        return [self.rid, self.start, self.n, self.max_len]

    @classmethod
    def from_list(cls, v):
        return cls(int(v[0]), int(v[1]), int(v[2]), int(v[3]))


class KVStore:
    """Per-request, per-layer K/V tensors on one worker."""

    def __init__(self, geom: ModelGeometry, device: str, dtype: torch.dtype):
        self.geom, self.device, self.dtype = geom, device, dtype
        self.caches: Dict[int, Dict[int, torch.Tensor]] = {}

    def get(self, rid: int, layer: int, max_len: int) -> torch.Tensor:
        per_req = self.caches.setdefault(rid, {})
        t = per_req.get(layer)
        if t is None:
            t = torch.empty(2, self.geom.kv_heads, max_len, self.geom.head_dim,
                            device=self.device, dtype=self.dtype)
            per_req[layer] = t
        return t

    def free(self, rid: int):
        self.caches.pop(rid, None)

    def bytes_used(self) -> int:
        # called from the stats thread while the compute thread mutates the dicts: snapshot first
        return sum(t.numel() * t.element_size() for d in list(self.caches.values()) for t in list(d.values()))

    def __len__(self):
        return len(self.caches)


class StageRunner:
    """Runs a contiguous set of decoder layers over a packed micro-batch."""

    def __init__(self, cfg, device: str, dtype: torch.dtype):
        self.cfg = cfg
        self.geom = ModelGeometry.from_config(cfg)
        self.device, self.dtype = device, dtype
        self.rotary = make_rotary(cfg, device)
        self.kv = KVStore(self.geom, device, dtype)
        self.layers: Dict[int, torch.nn.Module] = {}

    def positions(self, segments: Sequence[Segment]) -> torch.Tensor:
        pos = [torch.arange(s.start, s.start + s.n) for s in segments]
        return torch.cat(pos).to(self.device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, segments: Sequence[Segment], layer_ids: Sequence[int]) -> torch.Tensor:
        x = x.to(self.device, self.dtype)
        pos = self.positions(segments)
        cos, sin = self.rotary(x, pos[None, :])
        cos, sin = cos[0][:, None, :], sin[0][:, None, :]  # [T, 1, hd]
        for li in layer_ids:
            x = self._layer(self.layers[li], li, x, segments, cos, sin)
        return x

    def _layer(self, layer, li, x, segments, cos, sin):
        g = self.geom
        T = x.shape[0]
        attn = layer.self_attn
        h = layer.input_layernorm(x)
        q = attn.q_proj(h).view(T, g.heads, g.head_dim)
        k = attn.k_proj(h).view(T, g.kv_heads, g.head_dim)
        v = attn.v_proj(h).view(T, g.kv_heads, g.head_dim)
        if hasattr(attn, "q_norm"):  # Qwen3-style per-head norms
            q, k = attn.q_norm(q), attn.k_norm(k)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        out = torch.empty_like(q)
        off = 0
        gqa = g.heads != g.kv_heads
        for s in segments:
            cache = self.kv.get(s.rid, li, s.max_len)
            end = s.start + s.n
            cache[0, :, s.start:end] = k[off:off + s.n].transpose(0, 1)
            cache[1, :, s.start:end] = v[off:off + s.n].transpose(0, 1)
            qs = q[off:off + s.n].transpose(0, 1)[None]
            ks, vs = cache[0, :, :end][None], cache[1, :, :end][None]
            if s.n == 1:
                o = _sdpa(qs, ks, vs, gqa)
            elif s.start == 0:
                o = _sdpa(qs, ks, vs, gqa, is_causal=True)
            else:  # chunked prefill continuing an existing cache
                qi = torch.arange(s.start, end, device=x.device)[:, None]
                kj = torch.arange(end, device=x.device)[None, :]
                o = _sdpa(qs, ks, vs, gqa, attn_mask=kj <= qi)
            out[off:off + s.n] = o[0].transpose(0, 1)
            off += s.n
        x = x + attn.o_proj(out.reshape(T, g.heads * g.head_dim))
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        return x
