import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# _PadMatcher: a callable whose __ne__ returns False for ANY callable
# with __name__ == 'pad'.  This lets the SubgraphMatcher succeed even when
# Dynamo stores a different Python object for F.pad than FX symbolic_trace.
#
# __call__ creates the FX call_function node manually via the Proxy's tracer
# so that FX tracing of the pattern function works correctly.
# ──────────────────────────────────────────────────────────────────────────────
class _PadMatcher:
    __name__ = 'pad'

    def __eq__(self, other):
        return getattr(other, '__name__', '') == 'pad'

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash('pad')

    def __call__(self, input_proxy, *args, **kwargs):
        # During FX pattern tracing, input_proxy is a torch.fx.Proxy.
        # Manually ask the tracer to create a call_function node so that
        # the node's .target is exactly this _PadMatcher instance.
        if hasattr(input_proxy, 'tracer'):
            return input_proxy.tracer.create_proxy(
                'call_function',
                self,
                (input_proxy,) + args,
                kwargs,
            )
        raise RuntimeError("_PadMatcher invoked outside FX tracing context")


_PAD_MATCHER = _PadMatcher()


# ──────────────────────────────────────────────────────────────────────────────
# Pattern to match
# ──────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1):
    """
    Match:
      emb = embedding(in_0, in_1, 0, None, 2.0, False, False)   [B, S, D]
      s1  = emb[:, 1:]                                           [B, S-1, D]
      nxt = pad(s1, [0,0,0,1,0,0], 'constant', 0.0)             [B, S, D]
      s2  = emb[:, :-1]                                          [B, S-1, D]
      prv = pad(s2, [0,0,1,0,0,0], 'constant', 0.0)             [B, S, D]
      out = cat([nxt, emb, prv], dim=2)                          [B, S, 3D]

    _PAD_MATCHER matches any callable with __name__ == 'pad' so that the
    pattern works regardless of which exact pad function Dynamo records.
    """
    tmp_2 = torch.nn.functional.embedding(in_0, in_1, 0, None, 2.0, False, False)
    tmp_3 = tmp_2[(slice(None, None, None), slice(1, None, None))]
    tmp_4 = _PAD_MATCHER(tmp_3, [0, 0, 0, 1, 0, 0], 'constant', 0.0)
    tmp_5 = tmp_2[(slice(None, None, None), slice(None, -1, None))]
    tmp_6 = _PAD_MATCHER(tmp_5, [0, 0, 1, 0, 0, 0], 'constant', 0.0)
    tmp_7 = torch.cat([tmp_4, tmp_2, tmp_6], dim=2)
    return tmp_7


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel  (v3 – 1 scatter read, 3 writes, no external zeroing)
#
# Each program (b, s) reads embedding[in_0[b, s]] ONCE and writes it into
# three output slots.  Boundary zero slots are written by the boundary
# programs themselves, so we can use torch.empty for the output.
#
# Slot assignment (no races – every slot is written by exactly 1 program):
#   out[b, s,   D:2D]  ← curr slot  – always by program s
#   out[b, s-1, 0:D ]  ← prev pos's "next" slot – by program s   (s≥1)
#   out[b, s+1, 2D:3D] ← next pos's "prev" slot – by program s   (s+1<S)
#   out[b, 0,   2D:3D] ← zeros (no prev for s=0) – by program s=0
#   out[b, S-1, 0:D ]  ← zeros (no next for s=S-1) – by program s=S-1
# ──────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_D': 128}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_D': 128}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_D': 128}, num_warps=1, num_stages=1),
    ],
    key=['S', 'D'],
)
@triton.jit
def embedding_shift_cat_kernel(
    indices_ptr,   # int64  [B, S]   – contiguous
    weight_ptr,    # dtype  [V, D]   – contiguous
    out_ptr,       # dtype  [B, S, 3*D]  – uninitialised (torch.empty)
    B, S, D,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    b   = pid // S
    s   = pid  % S

    d_off  = tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    # ── Single scatter read from the embedding table ──────────────────────────
    idx = tl.load(indices_ptr + b * S + s)
    emb = tl.load(weight_ptr + idx * D + d_off, mask=d_mask, other=0.0)

    row    = b * S + s
    stride = 3 * D

    # ── Always: write curr slot ───────────────────────────────────────────────
    tl.store(out_ptr + row * stride + D + d_off, emb, mask=d_mask)

    # ── Write prev-position's "next" slot; or zero own "prev" slot at s==0 ───
    if s >= 1:
        # out[b, s-1, 0:D] = emb   (embedding at s is the "next" of s-1)
        tl.store(out_ptr + (row - 1) * stride + d_off, emb, mask=d_mask)
    else:
        # out[b, 0, 2D:3D] = 0     (position 0 has no predecessor)
        zero = tl.zeros([BLOCK_D], dtype=emb.dtype)
        tl.store(out_ptr + row * stride + 2 * D + d_off, zero, mask=d_mask)

    # ── Write next-position's "prev" slot; or zero own "next" slot at s==S-1 ─
    if s + 1 < S:
        # out[b, s+1, 2D:3D] = emb  (embedding at s is the "prev" of s+1)
        tl.store(out_ptr + (row + 1) * stride + 2 * D + d_off, emb, mask=d_mask)
    else:
        # out[b, S-1, 0:D] = 0      (last position has no successor)
        zero = tl.zeros([BLOCK_D], dtype=emb.dtype)
        tl.store(out_ptr + row * stride + d_off, zero, mask=d_mask)


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper (must be decorated with @torch.fx.wrap)
# ──────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def embedding_shift_cat(indices, weight):
    B, S = indices.shape
    V, D = weight.shape

    indices = indices.contiguous()
    weight  = weight.contiguous()

    # Use torch.empty – every slot is written by exactly one kernel program.
    out = torch.empty(B, S, 3 * D, dtype=weight.dtype, device=weight.device)

    embedding_shift_cat_kernel[(B * S,)](
        indices, weight, out,
        B, S, D,
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Replacement hook
# ──────────────────────────────────────────────────────────────────────────────
def replacement_func():
    return embedding_shift_cat