import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    tmp_1 = tmp_0.reshape(1, 124, 2, 768)
    tmp_2 = tmp_1.reshape(1, 248, 768)
    tmp_3 = torch.nn.functional.pad(tmp_2, (0, 0, 0, 1), 'constant', None)
    return tmp_3


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['n_total_elements'],
)
@triton.jit
def gelu_reshape_pad_kernel(
    x_ptr,
    out_ptr,
    n_gelu_elements,
    n_total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask_gelu = offsets < n_gelu_elements
    mask_total = offsets < n_total_elements
    mask_pad = (offsets >= n_gelu_elements) & mask_total

    # Load input and compute GELU for valid elements
    x = tl.load(x_ptr + offsets, mask=mask_gelu, other=0.0)
    x_fp32 = x.to(tl.float32)

    # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    sqrt_half: tl.constexpr = 0.7071067811865476
    result_fp32 = x_fp32 * 0.5 * (1.0 + tl.math.erf(x_fp32 * sqrt_half))
    result = result_fp32.to(x.dtype)

    # Store GELU results
    tl.store(out_ptr + offsets, result, mask=mask_gelu)

    # Store zeros for padding region
    zero = tl.zeros([BLOCK_SIZE], dtype=x.dtype)
    tl.store(out_ptr + offsets, zero, mask=mask_pad)


@torch.fx.wrap
def gelu_reshape_pad(in_0):
    # Input shape: [1, 124, 1536] -> 190464 elements
    # Output shape: [1, 249, 768] -> 191232 elements
    n_gelu_elements = 190464  # 1 * 124 * 1536 = 1 * 248 * 768
    n_total_elements = 191232  # 1 * 249 * 768

    out = torch.empty(1, 249, 768, dtype=in_0.dtype, device=in_0.device)

    BLOCK_SIZE = 1024  # placeholder, autotuning will pick the best
    num_programs = (n_total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    gelu_reshape_pad_kernel[(num_programs,)](
        in_0,
        out,
        n_gelu_elements,
        n_total_elements,
    )

    return out


def replacement_func():
    return gelu_reshape_pad