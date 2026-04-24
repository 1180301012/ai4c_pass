import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: in_3 += in_0; in_4 = in_3; in_4 += in_2; relu(tmp_0, inplace=True)
# Matches the iadd+iadd+relu chain on tensors of shape [1, 128, 16, 12].
# ---------------------------------------------------------------------------

def pattern(in_0, in_2, in_3):
    in_3 += in_0
    in_4 = in_3
    in_4 += in_2
    tmp_0 = in_4
    tmp_2 = torch.nn.functional.relu(tmp_0, inplace=True)
    return tmp_2


def replacement_args(in_0, in_2, in_3):
    return (in_0, in_2, in_3)


# ---------------------------------------------------------------------------
# Triton kernel: fused add + add + ReLU in a single memory pass.
# Each thread-block processes BLOCK_SIZE contiguous elements.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _fused_add_add_relu_kernel(
    x_ptr,        # in_0, shape [1, 128, 16, 12]
    y_ptr,        # in_2, shape [1, 128, 16, 12]
    z_ptr,        # in_3, shape [1, 128, 16, 12]  (read + write in-place)
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)

    # (z + x) + y, then ReLU — all in one shot
    result = tl.maximum(x + y + z, 0.0)

    tl.store(z_ptr + offsets, result, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper — called by the FX replacement pass.
# Reads in_0, in_2, in_3; writes result back to in_3's storage (in-place).
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_add_add_relu(in_0, in_2, in_3):
    N = in_3.numel()
    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    _fused_add_add_relu_kernel[grid](
        in_0,   # x_ptr
        in_2,   # y_ptr
        in_3,   # z_ptr  (read and written)
        N,
    )
    return in_3


def replacement_func():
    return fused_add_add_relu