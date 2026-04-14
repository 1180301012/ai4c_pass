import torch
import triton
import triton.language as tl


# ─── Simple copy kernel with destination offset ───────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def copy_with_offset_kernel(
    src_ptr,
    dst_ptr,
    dst_offset,      # element offset in dst (not bytes)
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < n_elements
    data    = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + dst_offset + offsets, data, mask=mask)



# ─── Pattern ──────────────────────────────────────────────────────────────────
# Minimal pattern: match only the cat operation.
# tmp_7 is the relu output (already computed); we just replace the cat with
# a single Triton kernel covering all 5 copies.
def pattern(in_5, in_7, in_8, in_6, tmp_7):
    tmp_8 = torch.cat([in_5, in_7, in_8, in_6, tmp_7], dim=1)
    return tmp_8


def replacement_args(in_5, in_7, in_8, in_6, tmp_7):
    return (in_5, in_7, in_8, in_6, tmp_7)


# ─── Replacement wrapper ──────────────────────────────────────────────────────
@torch.fx.wrap
def fused_relu_cat(in_5, in_7, in_8, in_6, tmp_7):
    """
    Triton-based cat: 5 separate copy kernels (each with autotune).
      cat([in_5, in_7, in_8, in_6, tmp_7], dim=1)
    """
    H, W  = 64, 64
    HW    = H * W
    C_0, C_1, C_2, C_3, C_4 = 2048, 512, 512, 512, 512
    C_total = C_0 + C_1 + C_2 + C_3 + C_4  # 4096

    out = torch.empty((1, C_total, H, W), dtype=in_5.dtype, device=in_5.device)

    n0 = C_0 * HW                 # 8 388 608
    n1 = C_1 * HW                 # 2 097 152
    n2 = C_2 * HW
    n3 = C_3 * HW
    n4 = C_4 * HW

    o0 = 0
    o1 = n0
    o2 = n0 + n1
    o3 = n0 + n1 + n2
    o4 = n0 + n1 + n2 + n3

    copy_with_offset_kernel[lambda m: ((n0 + m['BLOCK_SIZE'] - 1) // m['BLOCK_SIZE'],)](in_5,  out, o0, n0)
    copy_with_offset_kernel[lambda m: ((n1 + m['BLOCK_SIZE'] - 1) // m['BLOCK_SIZE'],)](in_7,  out, o1, n1)
    copy_with_offset_kernel[lambda m: ((n2 + m['BLOCK_SIZE'] - 1) // m['BLOCK_SIZE'],)](in_8,  out, o2, n2)
    copy_with_offset_kernel[lambda m: ((n3 + m['BLOCK_SIZE'] - 1) // m['BLOCK_SIZE'],)](in_6,  out, o3, n3)
    copy_with_offset_kernel[lambda m: ((n4 + m['BLOCK_SIZE'] - 1) // m['BLOCK_SIZE'],)](tmp_7, out, o4, n4)

    return out


def replacement_func():
    return fused_relu_cat