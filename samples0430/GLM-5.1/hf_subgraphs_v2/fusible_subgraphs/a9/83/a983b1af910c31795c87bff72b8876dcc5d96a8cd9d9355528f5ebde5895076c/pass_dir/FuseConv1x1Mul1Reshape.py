import torch
import triton
import triton.language as tl


def pattern(bias, weight, input):
    conv2d = torch.conv2d(input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d * 1.0
    tmp_4 = tmp_3.reshape(-1, 17, 4096)
    return (tmp_4,)


def replacement_args(bias, weight, input):
    return (bias, weight, input)


def _get_dtype_encoding(dtype):
    if dtype == torch.float32:
        return 0
    elif dtype == torch.float16:
        return 1
    elif dtype == torch.bfloat16:
        return 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


# Kernel v3: Row-wise computation with contiguous W access pattern
# This restructures the matmul so that input is loaded as [BLOCK_K, BLOCK_W]
# where W (spatial width) is contiguous (stride=1), providing much better
# memory coalescing than the original [BLOCK_M, BLOCK_K] with scattered K (stride=4096)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_CO': 16, 'BLOCK_K': 32, 'BLOCK_W': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_CO': 16, 'BLOCK_K': 64, 'BLOCK_W': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_CO': 32, 'BLOCK_K': 32, 'BLOCK_W': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_CO': 32, 'BLOCK_K': 64, 'BLOCK_W': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_CO': 16, 'BLOCK_K': 32, 'BLOCK_W': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_CO': 16, 'BLOCK_K': 64, 'BLOCK_W': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_CO': 32, 'BLOCK_K': 32, 'BLOCK_W': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_CO': 32, 'BLOCK_K': 64, 'BLOCK_W': 32}, num_stages=3, num_warps=8),
    ],
    key=['C_in', 'C_out', 'W_val', 'DTYPE_OUTPUT'],
)
@triton.jit
def fused_conv1x1_reshape_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N_batch, C_in, H, W_val, C_out, HW,
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,
    stride_w_co, stride_w_ci,
    stride_out_0, stride_out_1, stride_out_2,
    DTYPE_OUTPUT: tl.constexpr,
    BLOCK_CO: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_W: tl.constexpr,
):
    """
    Fused 1x1 Conv2D + identity multiply + reshape kernel.
    
    Uses row-wise computation with contiguous W access for better memory coalescing.
    
    For each (batch, h_row, co_block), computes:
    output[n, co:co+BLOCK_CO, h*W+w] = weight[co:co+BLOCK_CO, ci] @ input[n, ci, h, w:w+BLOCK_W] + bias[co]
    
    The input is loaded as [BLOCK_K, BLOCK_W] where W is contiguous (stride=1),
    providing efficient memory access compared to the NCHW scattered K access.
    """
    pid_batch = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_co = tl.program_id(2)
    
    n_idx = pid_batch
    h_idx = pid_h
    co_start = pid_co * BLOCK_CO
    
    co_offsets = co_start + tl.arange(0, BLOCK_CO)
    w_offsets = tl.arange(0, BLOCK_W)
    
    co_mask = co_offsets < C_out
    w_mask = w_offsets < W_val
    
    # Accumulator in float32 for precision: [BLOCK_CO, BLOCK_W]
    acc = tl.zeros((BLOCK_CO, BLOCK_W), dtype=tl.float32)
    
    # Pre-compute base offset for input (fixed n and h)
    input_base = n_idx * stride_in_n + h_idx * stride_in_h
    
    # Loop over K (C_in) dimension in blocks
    for k_start in range(0, C_in, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < C_in
        
        # Load weight[co, ci, 0, 0] as [BLOCK_CO, BLOCK_K] for tl.dot
        # weight is [C_out, C_in, 1, 1] - contiguous in C_in (stride_w_ci = 1)
        weight_offsets = (
            co_offsets[:, None] * stride_w_co + 
            k_offsets[None, :] * stride_w_ci
        )
        a = tl.load(weight_ptr + weight_offsets, mask=co_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load input[n, ci, h, w] as [BLOCK_K, BLOCK_W] for tl.dot
        # input is [N, C_in, H, W] - W is contiguous (stride_in_w = 1)
        # This access pattern is cache-friendly: within each K row, W values are contiguous
        input_offsets = (
            k_offsets[:, None] * stride_in_c + 
            w_offsets[None, :] * stride_in_w
        )
        b = tl.load(input_ptr + input_base + input_offsets, mask=k_mask[:, None] & w_mask[None, :], other=0.0)
        
        # tl.dot(a, b): [BLOCK_CO, BLOCK_K] @ [BLOCK_K, BLOCK_W] -> [BLOCK_CO, BLOCK_W]
        acc += tl.dot(a, b)
    
    # Add bias (cast to float32 for accumulation)
    bias_vals = tl.load(bias_ptr + co_offsets, mask=co_mask, other=0.0).to(tl.float32)
    acc += bias_vals[:, None]
    
    # Cast accumulator to output dtype before storing
    if DTYPE_OUTPUT == 1:  # float16
        acc = acc.to(tl.float16)
    elif DTYPE_OUTPUT == 2:  # bfloat16
        acc = acc.to(tl.bfloat16)
    # else: float32 (DTYPE_OUTPUT == 0), no cast needed
    
    # Store output[n, co, h*W + w] as 3D tensor [N, C_out, HW]
    hw_offsets = h_idx * W_val + w_offsets
    output_offsets = (
        n_idx * stride_out_0 + 
        co_offsets[:, None] * stride_out_1 + 
        hw_offsets[None, :] * stride_out_2
    )
    tl.store(output_ptr + output_offsets, acc, mask=co_mask[:, None] & w_mask[None, :])


@torch.fx.wrap
def fused_conv1x1_reshape(bias, weight, input_tensor):
    """
    Fused kernel that replaces:
    1. torch.conv2d(input, weight, bias, (1,1), (0,0), (1,1), 1) - 1x1 convolution
    2. result * 1.0 - identity multiply
    3. result.reshape(-1, 17, 4096) - reshape to target shape
    
    Uses row-wise computation with contiguous W memory access pattern.
    Grid: (N_batch, H, num_co_blocks) - 3D grid for parallelism.
    """
    N_batch = input_tensor.shape[0]
    C_in = input_tensor.shape[1]
    H = input_tensor.shape[2]
    W_val = input_tensor.shape[3]
    C_out = weight.shape[0]
    HW = H * W_val
    
    # Create output tensor in the target reshape shape
    output = torch.empty((N_batch, C_out, HW), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Get strides for proper indexing
    stride_in_n, stride_in_c, stride_in_h, stride_in_w = input_tensor.stride()
    stride_w_co, stride_w_ci, stride_w_1, stride_w_2 = weight.stride()
    stride_out_0, stride_out_1, stride_out_2 = output.stride()
    
    dtype_output = _get_dtype_encoding(output.dtype)
    
    # 3D grid: (N_batch, H, num_co_blocks)
    grid = lambda meta: (
        N_batch,
        H,
        triton.cdiv(C_out, meta['BLOCK_CO']),
    )
    
    fused_conv1x1_reshape_kernel[grid](
        input_ptr=input_tensor, weight_ptr=weight, bias_ptr=bias, output_ptr=output,
        N_batch=N_batch, C_in=C_in, H=H, W_val=W_val, C_out=C_out, HW=HW,
        stride_in_n=stride_in_n, stride_in_c=stride_in_c, stride_in_h=stride_in_h, stride_in_w=stride_in_w,
        stride_w_co=stride_w_co, stride_w_ci=stride_w_ci,
        stride_out_0=stride_out_0, stride_out_1=stride_out_1, stride_out_2=stride_out_2,
        DTYPE_OUTPUT=dtype_output,
    )
    
    return output


def replacement_func():
    return fused_conv1x1_reshape