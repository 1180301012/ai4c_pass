"""
Shared autotuned Triton kernel:
  fused(input[B, C, 256], output[B, C, 128, 128])

Each CTA owns one (b, c) pair and computes BLOCK_NC output positions along the
output spatial dimension.  For each output pixel it performs 4 coalesced
vector loads from the source tensor, accumulates in fp32, then stores the result.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32,  'BLOCK_NC': 512},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_C': 64,  'BLOCK_NC': 512},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_C': 128, 'BLOCK_NC': 512},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_C': 128, 'BLOCK_NC': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_C': 256, 'BLOCK_NC': 512},  num_warps=8, num_stages=4),
    ],
    key=['B', 'C', 'N'],
)
@triton.jit
def _fused_perm_bilinear_kernel(
    in_ptr,                  # [B, C, N]  – flattened linear output
    out_ptr,                 # [B, C, H2, W2]  – contiguous
    B, C, N,                 # N = 256 (spatial), H2=W2=128
    BLOCK_C: tl.constexpr,  # channels processed per CTA
    BLOCK_NC: tl.constexpr, # output spatial positions per CTA
):
    pid      = tl.program_id(0)
    c_blk    = tl.cdiv(C, BLOCK_C)

    bn      = pid // c_blk
    c_group = pid %  c_blk

    b       = bn     // C
    c_start = c_group * BLOCK_C

    # channel indices for this group
    c_off = c_start + tl.arange(0, BLOCK_C)    # [BLOCK_C]
    mask_c = c_off < C

    # Local spatial index within the N=256 source dimension
    n_local = tl.arange(0, BLOCK_NC)            # [BLOCK_NC]

    # bilinear source map: out[b, c, h, w]  ← in[b, c, s_h*256 + s_w]
    h2 = n_local // 128                    # output row  [BLOCK_NC]
    w2 = n_local  % 128                    # output col

    hs = h2.to(tl.float32) * (256.0 / 128.0) - 0.5
    ws = w2.to(tl.float32) * (256.0 / 128.0) - 0.5

    sh   = tl.floor(hs).to(tl.int32)
    sw   = tl.floor(ws).to(tl.int32)
    fh   = hs - sh.to(tl.float32)         # ✓
    fw   = ws - sw.to(tl.float32)         # ✓
    h0i  = tl.maximum(0, sh - 1)
    w0i  = tl.maximum(0, sw - 1)
    h1i  = tl.minimum(255, sh + 1)
    w1i  = tl.minimum(255, sw + 1)

    # memory offsets for the 4 bilinear corners (indexed into in[b, c, :])
    offs00 = h0i * 256 + w0i
    offs01 = h0i * 256 + w1i
    offs10 = h1i * 256 + w0i
    offs11 = h1i * 256 + w1i

    # base pointer for in[b, c_start, 0]
    base = b * C * 256 + c_start * 256

    acc = tl.zeros([BLOCK_NC], dtype=tl.float32)

    # four loads, each fp32 upcast
    acc += tl.load(in_ptr + base + c_off[:, None] * 256 + offs00[None, :],
                   mask=mask_c[:, None], other=0.0).to(tl.float32) * (
                   (1.0 - fh) * (1.0 - fw))
    acc += tl.load(in_ptr + base + c_off[:, None] * 256 + offs01[None, :],
                   mask=mask_c[:, None], other=0.0).to(tl.float32) * (
                   (1.0 - fh) * fw)
    acc += tl.load(in_ptr + base + c_off[:, None] * 256 + offs10[None, :],
                   mask=mask_c[:, None], other=0.0).to(tl.float32) * (
                   fh * (1.0 - fw))
    acc += tl.load(in_ptr + base + c_off[:, None] * 256 + offs11[None, :],
                   mask=mask_c[:, None], other=0.0).to(tl.float32) * fh * fw

    # output layout: out[b, c, h2, w2]  stride [C*128*128, 128*128, 128, 1]
    out_offs = b * (C * 128 * 128) + c_off[:, None] * 128 * 128 + h2[None, :] * 128 + w2[None, :]

    tl.store(out_ptr + out_offs,
             acc.to(out_ptr.dtype.element_ty),
             mask=mask_c[:, None])


def fused_perm_bilinear(in_t, B, C, N_in, H2, W2):
    """
    in_t  : [B, C, N_in]  contiguous  (equals the linear output)
    out   : [B, C, H2, W2] contiguous
    """
    out = torch.empty((B, C, H2, W2), dtype=in_t.dtype, device=in_t.device)

    grid = lambda meta: (B * triton.cdiv(C, meta['BLOCK_C']),)
    _fused_perm_bilinear_kernel[grid](
        in_t, out,
        B, C, N_in,
    )
    return out