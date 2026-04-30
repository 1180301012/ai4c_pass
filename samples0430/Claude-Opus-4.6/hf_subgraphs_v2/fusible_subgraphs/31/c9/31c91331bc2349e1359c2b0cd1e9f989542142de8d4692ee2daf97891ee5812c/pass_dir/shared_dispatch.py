import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_add_kernel(
    in_2_ptr, in_3_ptr, out_ptr,
    N, C: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N
    c_offsets = tl.arange(0, C)

    # in_2: [1, C, N], element [0,c,n] at c*N+n
    # in_3: [1, N, C], element [0,n,c] at n*C+c
    # out:  [1, N, C], element [0,n,c] at n*C+c
    in_2_idx = c_offsets[None, :] * N + n_offsets[:, None]
    in_3_idx = n_offsets[:, None] * C + c_offsets[None, :]

    x = tl.load(in_2_ptr + in_2_idx, mask=n_mask[:, None], other=0.0)
    x_f32 = x.to(tl.float32)
    gelu_x = x_f32 * 0.5 * (1.0 + tl.math.erf(x_f32 * 0.7071067811865476))

    residual = tl.load(in_3_ptr + in_3_idx, mask=n_mask[:, None], other=0.0)
    residual_f32 = residual.to(tl.float32)

    result = (residual_f32 + gelu_x).to(x.dtype)
    tl.store(out_ptr + in_3_idx, result, mask=n_mask[:, None])


@triton.jit
def _layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C: tl.constexpr,
):
    row_idx = tl.program_id(0)
    c_offsets = tl.arange(0, C)

    x = tl.load(x_ptr + row_idx * C + c_offsets)
    x_f32 = x.to(tl.float32)

    mean = tl.sum(x_f32, axis=0) / C
    diff = x_f32 - mean
    var = tl.sum(diff * diff, axis=0) / C
    inv_std = 1.0 / tl.sqrt(var + 1e-6)

    weight = tl.load(weight_ptr + c_offsets)
    bias = tl.load(bias_ptr + c_offsets)
    weight_f32 = weight.to(tl.float32)
    bias_f32 = bias.to(tl.float32)

    normed = weight_f32 * diff * inv_std + bias_f32
    tl.store(out_ptr + row_idx * C + c_offsets, normed.to(x.dtype))


@torch.fx.wrap
def fused_dispatch(*args):
    if len(args) == 2:
        # gelu+add case: args = (in_2, in_3)
        in_2, in_3 = args
        C = in_2.shape[1]
        H = in_2.shape[2]
        W = in_2.shape[3]
        N = H * W
        out = torch.empty(1, N, C, dtype=in_2.dtype, device=in_2.device)
        # Choose BLOCK_N based on N and C for good occupancy
        if C <= 32:
            BLOCK_N = 64
            nw = 4
        elif C <= 128:
            BLOCK_N = 16
            nw = 4
        else:
            BLOCK_N = 8
            nw = 4
        grid = ((N + BLOCK_N - 1) // BLOCK_N,)
        _gelu_add_kernel[grid](
            in_2, in_3, out,
            N=N, C=C, BLOCK_N=BLOCK_N, num_warps=nw,
        )
        return out
    else:
        # layer_norm+view case: args = (bias, weight, x, H, W)
        bias, weight, x, H, W = args[0], args[1], args[2], args[3], args[4]
        C = x.shape[2]
        N = x.shape[1]
        out = torch.empty(1, N, C, dtype=x.dtype, device=x.device)
        nw = 1 if C <= 32 else (4 if C <= 128 else 8)
        _layer_norm_kernel[(N,)](
            x, weight, bias, out,
            N=N, C=C, num_warps=nw,
        )
        return out.view(1, H, W, C)