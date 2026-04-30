import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.layer_norm(in_2, (512,), in_1, in_0, 1e-05)
    tmp_3 = tmp_2.transpose(-2, -1)
    tmp_4 = torch.nn.functional.gelu(tmp_3)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def layernorm_transpose_gelu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N, DN,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    n_start = pid_n * BLOCK_N

    n_offs = tl.arange(0, BLOCK_N)
    d_offs = tl.arange(0, BLOCK_D)

    n_mask = (n_start + n_offs) < N

    # Load BLOCK_N rows of BLOCK_D elements (coalesced reads)
    in_base = pid_b * N * BLOCK_D
    in_ptrs = in_base + (n_start + n_offs[:, None]) * BLOCK_D + d_offs[None, :]  # [BLOCK_N, BLOCK_D]
    x = tl.load(input_ptr + in_ptrs, mask=n_mask[:, None], other=0.0)
    x_fp32 = x.to(tl.float32)

    # Layer norm per row: reduce over D dimension (axis=1)
    mean = tl.sum(x_fp32, axis=1) / BLOCK_D  # [BLOCK_N]
    diff = x_fp32 - mean[:, None]  # [BLOCK_N, BLOCK_D]
    var = tl.sum(diff * diff, axis=1) / BLOCK_D  # [BLOCK_N]
    inv_std = 1.0 / tl.sqrt(var + 1e-05)  # [BLOCK_N]
    normalized = diff * inv_std[:, None]  # [BLOCK_N, BLOCK_D]

    # Affine transform
    w = tl.load(weight_ptr + d_offs).to(tl.float32)  # [BLOCK_D]
    bi = tl.load(bias_ptr + d_offs).to(tl.float32)  # [BLOCK_D]
    result = normalized * w[None, :] + bi[None, :]  # [BLOCK_N, BLOCK_D]

    # GELU (exact, erf-based)
    gelu_out = 0.5 * result * (1.0 + tl.math.erf(result * 0.7071067811865476))
    gelu_cast = gelu_out.to(x.dtype)  # [BLOCK_N, BLOCK_D]

    # Transpose for coalesced writes: output[b, d, n]
    result_t = tl.trans(gelu_cast)  # [BLOCK_D, BLOCK_N]
    out_base = pid_b * DN
    out_ptrs = out_base + d_offs[:, None] * N + (n_start + n_offs[None, :])  # [BLOCK_D, BLOCK_N]
    tl.store(output_ptr + out_ptrs, result_t, mask=n_mask[None, :])


@torch.fx.wrap
def layernorm_transpose_gelu(in_0, in_1, in_2):
    # in_0: bias [D], in_1: weight [D], in_2: input [B, N, D]
    B = in_2.shape[0]
    N = in_2.shape[1]
    D = in_2.shape[2]

    BLOCK_N = 16

    out = torch.empty(B, D, N, dtype=in_2.dtype, device=in_2.device)

    num_n_blocks = (N + BLOCK_N - 1) // BLOCK_N
    grid = (num_n_blocks, B)
    layernorm_transpose_gelu_kernel[grid](
        in_2, in_1, in_0, out,
        N, D * N,
        BLOCK_D=D,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=1,
    )

    return out


def replacement_func():
    return layernorm_transpose_gelu