import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    tmp_4 = tmp_3 * conv2d
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# Fully fused kernel: 1x1 conv (as matmul) + hardtanh + element-wise multiply
# This avoids all intermediate memory writes and reshape operations
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_conv1x1_hardtanh_mul_kernel(
    # Pointers
    a_ptr, b_ptr, bias_ptr, star_ptr, out_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides for A (input): 4D strides [N_batch, C_in, H, W]
    stride_a_nbatch, stride_a_cin, stride_a_h, stride_a_w,
    # Strides for B (weight): 4D strides [C_out, C_in, 1, 1]
    stride_b_cout, stride_b_cin,
    # Strides for star: 4D strides [N_batch, C_out, H, W]
    stride_star_nbatch, stride_star_cout, stride_star_h, stride_star_w,
    # Strides for output: 4D strides [N_batch, C_out, H, W]
    stride_out_nbatch, stride_out_cout, stride_out_h, stride_out_w,
    # Dimensions
    N_batch, C_in, C_out, H, W,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Kernel for computing: out = hardtanh(star) * (conv1x1(input, weight, bias))
    
    The 1x1 conv is computed as a matmul where:
    - A is [M, K] = [N_batch*H*W, C_in] (input reshaped conceptually)
    - B is [K, N] = [C_in, C_out] (weight reshaped conceptually)
    - Result = A @ B^T + bias
    
    But we compute using the 4D strides directly to avoid reshape.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute block start indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Decompose offs_m into (nbatch_idx, h_idx, w_idx) for 4D access
    HW = H * W
    nbatch_idx = offs_m // HW
    hw_idx = offs_m % HW
    h_idx = hw_idx // W
    w_idx = hw_idx % W

    # Initialize accumulator for matmul
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Matmul: accumulate over K dimension (C_in)
    for k_offset in range(0, K, BLOCK_K):
        offs_k_cur = k_offset + offs_k
        
        # Compute proper offsets for 4D access of A (input tensor)
        a_offsets = nbatch_idx[:, None] * stride_a_nbatch + \
                    offs_k_cur[None, :] * stride_a_cin + \
                    h_idx[:, None] * stride_a_h + \
                    w_idx[:, None] * stride_a_w
        a_mask = (offs_m[:, None] < M) & (offs_k_cur[None, :] < K)
        a = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0)
        
        # Load B block [BLOCK_K, BLOCK_N] using 4D strides of weight
        b_offsets = offs_k_cur[:, None] * stride_b_cin + offs_n[None, :] * stride_b_cout
        b_mask = (offs_k_cur[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)
        
        # Accumulate using tl.dot
        acc += tl.dot(a, b)

    # Add bias: broadcast over M dimension
    bias_offsets = offs_n
    bias_mask = offs_n < N
    bias = tl.load(bias_ptr + bias_offsets, mask=bias_mask, other=0.0)
    acc += bias[None, :]

    # Load star tensor values for hardtanh using 4D strides
    star_offsets = nbatch_idx[:, None] * stride_star_nbatch + \
                   offs_n[None, :] * stride_star_cout + \
                   h_idx[:, None] * stride_star_h + \
                   w_idx[:, None] * stride_star_w
    star_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    star = tl.load(star_ptr + star_offsets, mask=star_mask, other=0.0)

    # Apply hardtanh to star values: clamp(star, 0, 6)
    star_clamped = tl.minimum(tl.maximum(star, 0.0), 6.0)

    # Multiply: star_clamped * conv_result
    result = star_clamped * acc

    # Store result using 4D strides
    out_offsets = nbatch_idx[:, None] * stride_out_nbatch + \
                  offs_n[None, :] * stride_out_cout + \
                  h_idx[:, None] * stride_out_h + \
                  w_idx[:, None] * stride_out_w
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + out_offsets, result, mask=out_mask)


@torch.fx.wrap
def fused_conv1x1_hardtanh_mul(bias, weight, input_tensor, star_tensor):
    """
    Fused 1x1 conv + hardtanh + multiply.
    
    1x1 conv = input @ weight.T + bias (as matmul with 4D strides)
    output = hardtanh(star) * conv_result
    
    input_tensor: [N_batch, C_in, H, W]
    weight: [C_out, C_in, 1, 1]
    bias: [C_out]
    star_tensor: [N_batch, C_out, H, W]
    output: [N_batch, C_out, H, W]
    """
    N_batch, C_in, H, W = input_tensor.shape
    C_out = weight.shape[0]
    M = N_batch * H * W
    K = C_in
    
    # Allocate output with same shape as star_tensor
    out = torch.empty_like(star_tensor)
    
    # Get strides from 4D tensors directly
    stride_a_nbatch, stride_a_cin, stride_a_h, stride_a_w = input_tensor.stride()
    stride_b_cout, stride_b_cin = weight.stride()[0], weight.stride()[1]
    stride_star_nbatch, stride_star_cout, stride_star_h, stride_star_w = star_tensor.stride()
    stride_out_nbatch, stride_out_cout, stride_out_h, stride_out_w = out.stride()
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(C_out, meta['BLOCK_N']))
    
    fused_conv1x1_hardtanh_mul_kernel[grid](
        a_ptr=input_tensor, b_ptr=weight,
        bias_ptr=bias, star_ptr=star_tensor, out_ptr=out,
        M=M, N=C_out, K=K,
        stride_a_nbatch=stride_a_nbatch, stride_a_cin=stride_a_cin,
        stride_a_h=stride_a_h, stride_a_w=stride_a_w,
        stride_b_cout=stride_b_cout, stride_b_cin=stride_b_cin,
        stride_star_nbatch=stride_star_nbatch, stride_star_cout=stride_star_cout,
        stride_star_h=stride_star_h, stride_star_w=stride_star_w,
        stride_out_nbatch=stride_out_nbatch, stride_out_cout=stride_out_cout,
        stride_out_h=stride_out_h, stride_out_w=stride_out_w,
        N_batch=N_batch, C_in=C_in, C_out=C_out, H=H, W=W,
    )
    
    return out


def replacement_func():
    return fused_conv1x1_hardtanh_mul