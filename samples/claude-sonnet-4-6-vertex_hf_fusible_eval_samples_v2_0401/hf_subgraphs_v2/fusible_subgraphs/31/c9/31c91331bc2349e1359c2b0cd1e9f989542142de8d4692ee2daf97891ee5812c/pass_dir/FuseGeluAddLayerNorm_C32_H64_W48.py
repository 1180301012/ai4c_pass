import torch
import triton
import triton.language as tl

# Pass 1 of 2: Fuse GELU + reshape chain → produces tmp_10 (single output)
# C=32, H=64, W=48, N=3072
# in_2: [1, 32, 64, 48]  strides [C*N, N, W, 1]
# in_3: [1, 3072, 32]
# out (tmp_10): [1, 3072, 32]

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['N_VAL', 'C_VAL'],
)
@triton.jit
def _gelu_add_kernel_C32_H64_W48(
    in2_ptr, in3_ptr,
    out_ptr,
    N_VAL: tl.constexpr,
    C_VAL: tl.constexpr,
):
    # Each program handles exactly one row n
    n = tl.program_id(0)
    c_range = tl.arange(0, C_VAL)

    # Load in_2[0, c, h, w] where h*W+w = n
    # in_2 strides: [C*N, N, W, 1]  => offset = c*N + n
    in2_raw = tl.load(in2_ptr + c_range * N_VAL + n)
    in2 = in2_raw.to(tl.float32)

    # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    gelu_out = in2 * 0.5 * (1.0 + tl.math.erf(in2 * 0.7071067811865476))

    # Load in_3[0, n, :] contiguous
    in3 = tl.load(in3_ptr + n * C_VAL + c_range).to(tl.float32)

    # Add: gelu(in_2 rearranged) + in_3
    x = gelu_out + in3

    # Store result (tmp_10)
    tl.store(out_ptr + n * C_VAL + c_range, x.to(in2_raw.dtype))


# Pattern matches the full chain from in_2,in_3 → tmp_10 (SINGLE output)
def pattern(in_2, in_3):
    tmp_2 = torch.nn.functional.gelu(in_2, approximate='none')
    tmp_3 = tmp_2.flatten(2)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_5 = tmp_4.contiguous()
    tmp_6 = in_3 + tmp_5
    tmp_7 = tmp_6.permute(0, 2, 1)
    tmp_8 = tmp_7.view(1, 32, 64, 48)
    tmp_9 = tmp_8.view(1, 32, -1)
    tmp_10 = tmp_9.permute(0, 2, 1)
    return tmp_10


def replacement_args(in_2, in_3):
    return (in_2, in_3)


@torch.fx.wrap
def _gelu_add_C32_H64_W48(in_2, in_3):
    N_val = 3072
    C_val = 32

    out = torch.empty(1, N_val, C_val, dtype=in_2.dtype, device=in_2.device)

    _gelu_add_kernel_C32_H64_W48[(N_val,)](
        in_2, in_3,
        out,
        N_VAL=N_val,
        C_VAL=C_val,
    )

    return out


def replacement_func():
    return _gelu_add_C32_H64_W48