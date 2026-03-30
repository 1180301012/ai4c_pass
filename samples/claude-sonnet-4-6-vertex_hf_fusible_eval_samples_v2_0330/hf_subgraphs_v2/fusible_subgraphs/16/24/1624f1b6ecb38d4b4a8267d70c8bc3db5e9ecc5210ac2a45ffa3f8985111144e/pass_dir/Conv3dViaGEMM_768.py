"""
Conv3dViaGEMM_768.py

Replaces:
    conv3d(x, weight, bias, (2,16,16), (0,0,0), (1,1,1), 1)
    .flatten(2)
    .transpose(1, 2)

with im2col (zero-copy view + permute) followed by cuBLAS matmul (@),
plus a Triton kernel for the fused bias broadcast-add.

For stride==kernel (our case), patches are non-overlapping, so im2col
reduces to a contiguous view + permute with NO data duplication.

Fixed geometry:
  Input   : (B, 3, 10, 224, 224)
  Weight  : (768, 3, 2, 16, 16)  → (N=768, K=1536) flattened
  Output  : (B, 980, 768)        [= tmp_8 after flatten+transpose]

GEMM: (M=980, K=1536) @ (K=1536, N=768) + bias
"""

import torch
import triton
import triton.language as tl


# ── Triton kernel: in-place broadcast-add of bias to each row ─────────────────

@triton.autotune(
    configs=[
        triton.Config({'CHUNK': 256}, num_warps=4,  num_stages=1),
        triton.Config({'CHUNK': 256}, num_warps=8,  num_stages=1),
        triton.Config({'CHUNK': 256}, num_warps=16, num_stages=1),
        triton.Config({'CHUNK': 512}, num_warps=4,  num_stages=1),
        triton.Config({'CHUNK': 512}, num_warps=8,  num_stages=1),
        triton.Config({'CHUNK': 1024}, num_warps=4, num_stages=1),
        triton.Config({'CHUNK': 1024}, num_warps=8, num_stages=1),
    ],
    key=['M'],
)
@triton.jit
def bias_add_inplace_kernel(
    out_ptr,
    bias_ptr,
    M,
    IS_BF16: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """
    out[row, col] += bias[col]  for all rows in [0, M).
    N=768 is hardcoded; one program per row.
    """
    N_COLS: tl.constexpr = 768
    N_ITERS: tl.constexpr = (N_COLS + CHUNK - 1) // CHUNK

    row  = tl.program_id(0)
    base = row * N_COLS

    for i in tl.static_range(N_ITERS):
        cols = i * CHUNK + tl.arange(0, CHUNK)
        mask = cols < N_COLS
        x = tl.load(out_ptr  + base + cols, mask=mask, other=0.0)
        b = tl.load(bias_ptr + cols,        mask=mask, other=0.0)
        tl.store(out_ptr + base + cols, x + b, mask=mask)


@torch.fx.wrap
def conv3d_via_gemm_impl(x, weight, bias):
    """
    Drop-in for conv3d(x,w,b,(2,16,16),(0,0,0),(1,1,1),1).flatten(2).transpose(1,2).

    Returns (B, M=980, N=768) matching tmp_8 in the original graph.

    Steps
    -----
    1. Zero-copy view: x → (B, Cin, Od, Kd, Oh, Kh, Ow, Kw)  [valid for stride==kernel]
    2. Contiguous permute: → (B, Od, Oh, Ow, Cin, Kd, Kh, Kw)
    3. Flat reshape: → (B*M, K) = (980, 1536)
    4. cuBLAS matmul (@): (980,1536) @ (1536,768) = (980,768)
    5. Triton bias add (in-place)
    6. View: → (B, M, N) = (1, 980, 768)
    """
    B, Cin, D, H, W = x.shape
    Cout, _, Kd, Kh, Kw = weight.shape
    Sd, Sh, Sw = 2, 16, 16

    Od = (D  - Kd) // Sd + 1   # = 5
    Oh = (H  - Kh) // Sh + 1   # = 14
    Ow = (W  - Kw) // Sw + 1   # = 14
    M  = Od * Oh * Ow           # = 980
    K  = Cin * Kd * Kh * Kw    # = 1536
    N  = Cout                   # = 768

    # Step 1: zero-copy reshape (valid because stride==kernel → non-overlapping patches)
    x_view = x.view(B, Cin, Od, Kd, Oh, Kh, Ow, Kw)

    # Step 2: permute to (B, Od, Oh, Ow, Cin, Kd, Kh, Kw) and make contiguous
    x_patches = x_view.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()

    # Step 3: flatten to (B*M, K)
    x_flat = x_patches.view(B * M, K)

    # Step 4: (B*M, K) @ (K, N) = (B*M, N) via cuBLAS (@ operator, not torch.mm)
    w_flat = weight.view(N, K)
    out    = x_flat @ w_flat.t()   # shape: (B*M, N) = (980, 768)

    # Step 5: Triton in-place bias broadcast-add
    IS_BF16 = (out.dtype == torch.bfloat16)
    bias_add_inplace_kernel[(B * M,)](
        out, bias, B * M,
        IS_BF16=IS_BF16,
    )

    # Step 6: reshape to (B, M, N) = (1, 980, 768)
    return out.view(B, M, N)


# ─── Pattern / replacement glue ──────────────────────────────────────────────

def pattern(in_6, in_1, in_0):
    """
    Matches:
        conv3d_out = torch.conv3d(in_6, in_1, in_0, (2,16,16), (0,0,0), (1,1,1), 1)
        flat       = conv3d_out.flatten(2)
        transposed = flat.transpose(1, 2)
        return transposed
    """
    conv3d_out = torch.conv3d(in_6, in_1, in_0, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1)
    flat       = conv3d_out.flatten(2)
    transposed = flat.transpose(1, 2)
    return transposed


def replacement_args(in_6, in_1, in_0):
    return (in_6, in_1, in_0)


def replacement_func():
    return conv3d_via_gemm_impl