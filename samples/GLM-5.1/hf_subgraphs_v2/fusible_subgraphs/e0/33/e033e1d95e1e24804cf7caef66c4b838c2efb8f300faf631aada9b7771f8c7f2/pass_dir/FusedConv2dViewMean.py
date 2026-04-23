import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(conv2d.shape[0], 256, -1)
    tmp_4 = in_2.mean(dim=-2, keepdim=True)
    return (tmp_4, tmp_3)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "conv2d_view_mean")

# Triton kernel for 1x1 conv2d + view fusion
# The 1x1 conv2d with no padding is equivalent to a linear/matrix multiplication
# Input: in_3 shape [B, C_in, H, W], weight: in_1 shape [C_out, C_in, 1, 1], bias: in_0 shape [C_out]
# Output: shape [B, C_out, H*W] (fused view)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_view_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, C_in, H, W, C_out,
    M, N, K,
    stride_input_b, stride_input_c, stride_input_h, stride_input_w,
    stride_weight_oc, stride_weight_ic,
    stride_output_b, stride_output_oc, stride_output_hw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # pid = tl.program_id(0)
    # grid is (BLOCK_M groups * BLOCK_N groups)
    # This is a matmul kernel: input reshaped as [B*H*W, C_in] @ weight reshaped as [C_in, C_out] + bias
    # Output shape: [B*H*W, C_out] stored as [B, C_out, H*W]
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # offsets for output rows (M dimension = B*H*W) and columns (N dimension = C_out)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Compute which batch/spatial position each row corresponds to
    # row_idx = b * H * W + hw
    HW = H * W
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_offset in range(0, K, BLOCK_K):
        offs_k = k_offset + tl.arange(0, BLOCK_K)
        
        # Load input: shape [B*H*W, C_in]
        # input[b, c, h, w] -> input_ptr + b * stride_input_b + c * stride_input_c + h * stride_input_h + w * stride_input_w
        # For row offs_m[i], compute b, hw, then h = hw // W, w = hw % W
        # input_idx = b * stride_input_b + offs_k[j] * stride_input_c + (offs_m[i] // W) * stride_input_h + (offs_m[i] % W) * stride_input_w
        
        b_idx = offs_m // HW  # batch index
        hw_idx = offs_m % HW  # spatial index
        h_idx = hw_idx // W
        w_idx = hw_idx % W
        
        input_ptrs = input_ptr + b_idx[:, None] * stride_input_b + offs_k[None, :] * stride_input_c + h_idx[:, None] * stride_input_h + w_idx[:, None] * stride_input_w
        
        mask_input_m = offs_m < M
        mask_input_k = offs_k < K
        mask_input = mask_input_m[:, None] & mask_input_k[None, :]
        
        a = tl.load(input_ptrs, mask=mask_input, other=0.0)
        
        # Load weight: shape [C_out, C_in, 1, 1] -> [C_out, C_in]
        # weight[c_out, c_in, 0, 0] = weight_ptr + c_out * stride_weight_oc + c_in * stride_weight_ic
        weight_ptrs = weight_ptr + offs_n[None, :] * stride_weight_oc + offs_k[:, None] * stride_weight_ic
        
        mask_weight_n = offs_n < N
        mask_weight_k = offs_k < K
        mask_weight = mask_weight_k[:, None] & mask_weight_n[None, :]
        
        b_weight = tl.load(weight_ptrs, mask=mask_weight, other=0.0)
        
        acc += tl.dot(a, b_weight, allow_tf32=True)
    
    # Add bias
    bias_ptrs = bias_ptr + offs_n[None, :]
    mask_bias_n = offs_n < N
    bias = tl.load(bias_ptrs, mask=mask_bias_n, other=0.0)
    acc += bias
    
    # Store output: shape [B, C_out, H*W]
    # output[b, c_out, hw] = output_ptr + b * stride_output_b + c_out * stride_output_oc + hw * stride_output_hw
    # For row offs_m[i], b = offs_m[i] // HW, hw = offs_m[i] % HW
    output_ptrs = output_ptr + b_idx[:, None] * stride_output_b + offs_n[None, :] * stride_output_oc + hw_idx[:, None] * stride_output_hw
    
    mask_output_m = offs_m < M
    mask_output_n = offs_n < N
    mask_output = mask_output_m[:, None] & mask_output_n[None, :]
    
    # Cast to output dtype
    output_dtype = output_ptr.dtype.element_ty
    tl.store(output_ptrs, acc.to(output_dtype), mask=mask_output)


# Triton kernel for mean(dim=-2, keepdim=True)
# Input shape: [B, D, C] where D is the dimension being reduced
# Output shape: [B, 1, C]

@triton.jit
def mean_kernel(
    input_ptr, output_ptr,
    B, D, C,
    stride_input_b, stride_input_d, stride_input_c,
    stride_output_b, stride_output_1, stride_output_c,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    
    acc = tl.zeros((BLOCK_D, BLOCK_C), dtype=tl.float32)
    
    for d_start in range(0, D, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        
        input_ptrs = input_ptr + pid_b * stride_input_b + offs_d[:, None] * stride_input_d + offs_c[None, :] * stride_input_c
        mask_load = mask_d[:, None] & mask_c[None, :]
        
        vals = tl.load(input_ptrs, mask=mask_load, other=0.0)
        acc += vals
    
    # Sum along D dimension and divide by D
    mean_vals = tl.sum(acc, axis=0) / D
    
    output_ptrs = output_ptr + pid_b * stride_output_b + 0 * stride_output_1 + offs_c * stride_output_c
    mask_store = mask_c
    
    output_dtype = output_ptr.dtype.element_ty
    tl.store(output_ptrs, mean_vals.to(output_dtype), mask=mask_store)


@torch.fx.wrap
def fused_conv2d_view_mean(in_0, in_1, in_2, in_3, route):
    if route == "conv2d_view_mean":
        # Compute conv2d + view
        B = in_3.shape[0]
        C_in = in_3.shape[1]
        H = in_3.shape[2]
        W = in_3.shape[3]
        C_out = in_1.shape[0]
        
        HW = H * W
        M = B * HW
        N = C_out
        K = C_in
        
        tmp_3 = torch.empty((B, C_out, HW), dtype=in_3.dtype, device=in_3.device)
        
        # Launch conv2d+view kernel
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        
        grid_m = (M + BLOCK_M - 1) // BLOCK_M
        grid_n = (N + BLOCK_N - 1) // BLOCK_N
        
        conv2d_view_kernel[(grid_m, grid_n)](
            in_3, in_1, in_0, tmp_3,
            B, C_in, H, W, C_out,
            M, N, K,
            in_3.stride(0), in_3.stride(1), in_3.stride(2), in_3.stride(3),
            in_1.stride(0), in_1.stride(1),
            tmp_3.stride(0), tmp_3.stride(1), tmp_3.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        
        # Compute mean
        D = in_2.shape[1]  # 4096
        C_mean = in_2.shape[2]  # 256
        B_mean = in_2.shape[0]
        
        tmp_4 = torch.empty((B_mean, 1, C_mean), dtype=in_2.dtype, device=in_2.device)
        
        BLOCK_C_MEAN = 64
        BLOCK_D_MEAN = 256
        
        grid_mean = (B_mean, (C_mean + BLOCK_C_MEAN - 1) // BLOCK_C_MEAN)
        
        mean_kernel[grid_mean](
            in_2, tmp_4,
            B_mean, D, C_mean,
            in_2.stride(0), in_2.stride(1), in_2.stride(2),
            tmp_4.stride(0), tmp_4.stride(1), tmp_4.stride(2),
            BLOCK_C=BLOCK_C_MEAN, BLOCK_D=BLOCK_D_MEAN,
        )
        
        return (tmp_4, tmp_3)
    else:
        raise ValueError(f"Unknown route: {route}")

def replacement_func():
    return fused_conv2d_view_mean