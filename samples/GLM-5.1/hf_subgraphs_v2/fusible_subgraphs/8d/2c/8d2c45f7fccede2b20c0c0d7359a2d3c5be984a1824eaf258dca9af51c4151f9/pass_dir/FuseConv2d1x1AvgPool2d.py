import torch
import triton
import triton.language as tl
import math

# Pattern: Conv2d (1x1, stride=1, padding=0, dilation=1, groups=1) followed by avg_pool2d (kernel=2, stride=2)
def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.avg_pool2d(conv2d, 2, 2, 0, False, True, None)
    return (tmp_2,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N_dim', 'K'],
)
@triton.jit
def fused_conv1x1_avgpool2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    M,              # N_batch * H_OUT * W_OUT
    N_dim,          # C_OUT
    K,              # C_IN
    H_IN,
    W_IN,
    H_OUT,
    W_OUT,
    N_batch,
    stride_n_in,
    stride_ci_in,
    stride_hi_in,
    stride_n_out,
    stride_co_out,
    stride_ho_out,
    stride_co_w,
    stride_ci_w,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Fused 1x1 Conv2d + AvgPool2d kernel.
    Computes: output = pooled_input @ weight.T
    where pooled_input is computed on-the-fly from the original input.
    Uses input dtype for tl.dot to leverage tensor cores.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N_dim, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode m into (n_batch_idx, ho, wo)
    n_batch_idx = offs_m // (H_OUT * W_OUT)
    hw_rem = offs_m % (H_OUT * W_OUT)
    ho = hw_rem // W_OUT
    wo = hw_rem % W_OUT

    mask_m = offs_m < M
    mask_n = offs_n < N_dim

    mask_n_batch = n_batch_idx < N_batch
    mask_ho_valid = ho < H_OUT
    mask_wo_valid = wo < W_OUT
    mask_spatial = mask_m & mask_n_batch & mask_ho_valid & mask_wo_valid

    # Accumulate in float32 for precision
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load weight.T tile: [BLOCK_K, BLOCK_N] in original dtype
        w_ptrs = offs_k[:, None] * stride_ci_w + offs_n[None, :] * stride_co_w
        mask_wt = mask_k[:, None] & mask_n[None, :]
        w_t = tl.load(weight_ptr + w_ptrs, mask=mask_wt, other=0.0)

        # Compute pooled input on-the-fly: [BLOCK_M, BLOCK_K]
        # Use float32 for pooling accumulation to avoid precision issues,
        # then convert back to input dtype for tl.dot
        pooled = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

        for dh in range(2):
            for dw in range(2):
                hi = ho * 2 + dh
                wi = wo * 2 + dw
                mask_hi = (hi < H_IN) & mask_spatial
                mask_wi = (wi < W_IN) & mask_spatial
                mask_load = mask_hi & mask_wi

                # input[n, ci, hi, wi] - load in original dtype, then upcast for pooling
                i_ptrs = n_batch_idx[:, None] * stride_n_in + \
                         offs_k[None, :] * stride_ci_in + \
                         hi[:, None] * stride_hi_in + \
                         wi[:, None]
                mask_i = mask_spatial[:, None] & mask_k[None, :]
                inp = tl.load(input_ptr + i_ptrs, mask=mask_i, other=0.0)
                pooled += inp.to(tl.float32)

        pooled = pooled * 0.25

        # Convert pooled back to input dtype for tl.dot with w_t
        # This allows tensor core usage for float16/bfloat16
        input_dtype = input_ptr.dtype.element_ty
        pooled_cast = pooled.to(input_dtype)

        # tl.dot(pooled_cast, w_t) - both in input dtype, leveraging tensor cores
        # Result upcasted to float32 via accumulator
        accumulator += tl.dot(pooled_cast, w_t, allow_tf32=True)

    # Store output in original dtype
    o_ptrs = n_batch_idx[:, None] * stride_n_out + \
             offs_n[None, :] * stride_co_out + \
             ho[:, None] * stride_ho_out + \
             wo[:, None]
    mask_o = mask_spatial[:, None] & mask_n[None, :]

    output_dtype = output_ptr.dtype.element_ty
    accumulator = accumulator.to(output_dtype)
    tl.store(output_ptr + o_ptrs, accumulator, mask=mask_o)


@torch.fx.wrap
def fused_conv1x1_avgpool2d(weight, input):
    # weight: [C_OUT, C_IN, 1, 1]
    # input: [N, C_IN, H_IN, W_IN]
    # output: [N, C_OUT, H_OUT, W_OUT] where H_OUT=H_IN//2, W_OUT=W_IN//2
    
    N_batch = input.shape[0]
    C_IN = input.shape[1]
    H_IN = input.shape[2]
    W_IN = input.shape[3]
    C_OUT = weight.shape[0]
    H_OUT = H_IN // 2
    W_OUT = W_IN // 2
    
    M = N_batch * H_OUT * W_OUT
    N_dim = C_OUT
    K = C_IN
    
    # Output tensor
    output = torch.empty((N_batch, C_OUT, H_OUT, W_OUT), dtype=input.dtype, device=input.device)
    
    # Strides
    stride_n_in = C_IN * H_IN * W_IN
    stride_ci_in = H_IN * W_IN
    stride_hi_in = W_IN
    
    stride_n_out = C_OUT * H_OUT * W_OUT
    stride_co_out = H_OUT * W_OUT
    stride_ho_out = W_OUT
    
    stride_co_w = C_IN
    stride_ci_w = 1
    
    # Grid: total programs adapts to chosen BLOCK sizes via autotune
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N_dim, META['BLOCK_N']),)
    
    fused_conv1x1_avgpool2d_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        output_ptr=output,
        M=M,
        N_dim=N_dim,
        K=K,
        H_IN=H_IN,
        W_IN=W_IN,
        H_OUT=H_OUT,
        W_OUT=W_OUT,
        N_batch=N_batch,
        stride_n_in=stride_n_in,
        stride_ci_in=stride_ci_in,
        stride_hi_in=stride_hi_in,
        stride_n_out=stride_n_out,
        stride_co_out=stride_co_out,
        stride_ho_out=stride_ho_out,
        stride_co_w=stride_co_w,
        stride_ci_w=stride_ci_w,
    )
    
    return output


def replacement_func():
    return fused_conv1x1_avgpool2d