import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_conv1x1_bn_add_kernel(
    input_ptr,   # [batch, C_in, H, W] - NCHW
    weight_ptr,  # [C_out, C_in]  (1x1 conv weight flattened)
    mean_ptr,    # [C_out] - BN running mean
    var_ptr,     # [C_out] - BN running var
    bn_w_ptr,    # [C_out] - BN affine weight (gamma)
    bn_b_ptr,    # [C_out] - BN affine bias (beta)
    residual_ptr, # [batch, C_out, H, W] - optional residual (None if not present)
    output_ptr,  # [batch, C_out, H, W]
    M, N, K,     # M = batch*H*W, N = C_out, K = C_in
    HW,          # H * W
    eps,
    DO_RESNET10T: tl.constexpr,  # 0 for deeppose (bn_out += res), 1 for resnet10t (res += bn_out)
    DTYPE: tl.constexpr,         # 0=float16, 1=bfloat16, 2=float32
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Decompose flat index m = batch*HW + s  (s = h*W + w)
    batch_idx = offs_m // HW
    s_idx = offs_m % HW

    # NCHW linear offsets:
    #   input  : batch * C_in * HW  +  c_in * HW  +  s
    #   output : batch * C_out * HW + c_out * HW  +  s
    #   residual: same as output
    input_base = batch_idx * (K * HW) + s_idx
    out_base   = batch_idx * (N * HW) + s_idx

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load input tile [BLOCK_M, BLOCK_K]
        inp_ptrs = input_ptr + input_base[:, None] + offs_k[None, :] * HW
        a = tl.load(inp_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load weight tile [BLOCK_K, BLOCK_N]  (weight is [C_out, C_in] → transpose)
        w_ptrs = weight_ptr + offs_n[None, :] * K + offs_k[:, None]
        b = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(a, b)

    # Load BN parameters [BLOCK_N]
    mean  = tl.load(mean_ptr  + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    var   = tl.load(var_ptr   + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    gamma = tl.load(bn_w_ptr  + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    beta  = tl.load(bn_b_ptr  + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    inv_std   = gamma / tl.sqrt(var + eps)
    bn_offset = beta - mean * inv_std
    acc = acc * inv_std[None, :] + bn_offset[None, :]

    out_ptrs = output_ptr + out_base[:, None] + offs_n[None, :] * HW

    if DO_RESNET10T:
        # resnet10t: residual += bn_out  →  output = residual + acc
        res = tl.load(residual_ptr + out_base[:, None] + offs_n[None, :] * HW,
                      mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
        acc = res + acc

    if DTYPE == 0:
        tl.store(out_ptrs, acc.to(tl.float16),   mask=mask_m[:, None] & mask_n[None, :])
    elif DTYPE == 1:
        tl.store(out_ptrs, acc.to(tl.bfloat16),  mask=mask_m[:, None] & mask_n[None, :])
    else:
        tl.store(out_ptrs, acc,                  mask=mask_m[:, None] & mask_n[None, :])


# Map dtype → int for constexpr
_DTYPE_MAP = {
    torch.float16:  0,
    torch.bfloat16: 1,
    torch.float32:  2,
}


def launch_fused_kernel(input_tensor, weight, running_mean, running_var,
                        bn_bias, bn_weight, residual, output_dtype):
    """Common launcher for both deeppose and resnet10t patterns."""
    batch_size = input_tensor.shape[0]
    C_in  = input_tensor.shape[1]
    H     = input_tensor.shape[2]
    W     = input_tensor.shape[3]
    C_out = weight.shape[0]
    HW    = H * W
    M     = batch_size * HW

    output = torch.empty_like(residual if residual is not None
                              else output_tensor)

    weight_2d = weight.view(C_out, C_in)

    dtype_id = _DTYPE_MAP.get(output_dtype, 2)
    has_res  = residual is not None

    grid = lambda meta: (
        triton.cdiv(M,     meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N']),
    )

    fused_conv1x1_bn_add_kernel[grid](
        input_tensor, weight_2d,
        running_mean, running_var, bn_weight, bn_bias,
        residual,
        output,
        M, C_out, C_in,
        HW,
        1e-5,
        DO_RESNET10T=1 if has_res else 0,
        DTYPE=dtype_id,
    )
    return output