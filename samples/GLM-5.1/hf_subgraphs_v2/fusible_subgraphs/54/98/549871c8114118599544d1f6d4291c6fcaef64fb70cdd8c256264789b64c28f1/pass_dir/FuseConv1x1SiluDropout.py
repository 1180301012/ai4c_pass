import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.silu(conv2d, inplace=False)
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_2, in_1, in_0)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_silu_kernel(
    X_ptr, W_ptr, B_ptr, O_ptr,
    M, N, K,
    H, W_sp,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wc,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute row indices and decompose into (n, h, w)
    row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    HW = H * W_sp
    n = row_idx // HW
    hw = row_idx % HW
    h = hw // W_sp
    w = hw % W_sp
    row_mask = row_idx < M

    # Compute column indices (c_out)
    c_out = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = c_out < N

    # Initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K (c_in dimension)
    for k_start in range(0, K, BLOCK_K):
        c_in = k_start + tl.arange(0, BLOCK_K)

        # Load input tile: input[n, c_in, h, w]
        input_base = n * stride_xn + h * stride_xh + w * stride_xw
        x_ptrs = X_ptr + input_base[:, None] + c_in[None, :] * stride_xc
        x_mask = row_mask[:, None] & (c_in[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load weight tile: weight[c_out, c_in, 0, 0] -> weight[c_out, c_in]
        w_ptrs = W_ptr + c_in[:, None] * stride_wc + c_out[None, :] * stride_wo
        w_mask = (c_in[:, None] < K) & col_mask[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate using tl.dot (uses tensor cores for bf16/fp16, tf32 for fp32)
        acc += tl.dot(x, w, allow_tf32=True)

    # Load bias: [BLOCK_N]
    b_ptrs = B_ptr + c_out
    b_mask = col_mask
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)
    acc += b[None, :]

    # SiLU activation: x * sigmoid(x)
    silu = acc * tl.sigmoid(acc)

    # Store output: output[n, c_out, h, w]
    output_base = n * stride_on + h * stride_oh + w * stride_ow
    o_ptrs = O_ptr + output_base[:, None] + c_out[None, :] * stride_oc
    o_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(o_ptrs, silu, mask=o_mask)


@torch.fx.wrap
def conv1x1_silu_fused(input, weight, bias):
    N_batch, C_in, H, W = input.shape
    C_out = weight.shape[0]
    M = N_batch * H * W

    output = torch.empty((N_batch, C_out, H, W), dtype=input.dtype, device=input.device)

    stride_xn, stride_xc, stride_xh, stride_xw = input.stride()
    stride_wo, stride_wc = weight.stride()[0], weight.stride()[1]
    stride_on, stride_oc, stride_oh, stride_ow = output.stride()

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(C_out, META['BLOCK_N']))

    conv1x1_silu_kernel[grid](
        input, weight, bias, output,
        M, C_out, C_in,
        H, W,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_wo, stride_wc,
        stride_on, stride_oc, stride_oh, stride_ow,
    )

    return output


def replacement_func():
    return conv1x1_silu_fused