import torch
import triton
import triton.language as tl

def pattern(bias, weight, input_tensor):
    """
    Pattern: 1x1 conv2d + multiply by 1.0 + reshape
    """
    conv_out = torch.conv2d(input_tensor, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    mul_out = conv_out * 1.0
    reshaped = mul_out.reshape(-1, 17, 4096)
    return (reshaped,)

def replacement_args(bias, weight, input_tensor):
    return (bias, weight, input_tensor)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=1, num_warps=8),
    ],
    key=['HW', 'C'],
)
@triton.jit
def conv1x1_nchw_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, C, HW,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel that works directly with NCHW input format.
    """
    pid = tl.program_id(0)
    
    # Total spatial positions across all batches
    total_hw = B * HW
    
    # Process BLOCK_SIZE spatial positions
    pos_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    pos_mask = pos_offs < total_hw
    
    # Compute batch index and spatial position within batch
    b_idx = pos_offs // HW
    spatial_idx = pos_offs % HW
    
    # Output channel offsets (N=17, pad to 32)
    n_offs = tl.arange(0, 32)
    n_mask = n_offs < N
    
    # Accumulator [BLOCK_SIZE, 32]
    acc = tl.zeros((BLOCK_SIZE, 32), dtype=tl.float32)
    
    # Loop over input channels
    for c in range(C):
        # Load input [BLOCK_SIZE] from NCHW format
        # input shape: [B, C, HW] at index [b, c, spatial]
        input_idx = b_idx * (C * HW) + c * HW + spatial_idx
        inp_val = tl.load(input_ptr + input_idx, mask=pos_mask, other=0.0)
        
        # Load weight [32] for this input channel
        # weight shape: [N, C] at index [n, c]
        weight_idx = n_offs * C + c
        wgt_val = tl.load(weight_ptr + weight_idx, mask=n_mask, other=0.0)
        
        # Accumulate
        acc += inp_val[:, None] * wgt_val[None, :]
    
    # Add bias
    bias_vals = tl.load(bias_ptr + n_offs, mask=n_mask, other=0.0)
    acc += bias_vals[None, :]
    
    # Store output [BLOCK_SIZE, N]
    # Output format: [total_hw, N]
    output_idx = pos_offs[:, None] * N + n_offs[None, :]
    output_mask = pos_mask[:, None] & n_mask[None, :]
    tl.store(output_ptr + output_idx, acc, mask=output_mask)

@torch.fx.wrap
def fused_conv1x1_mul_reshape(bias, weight, input_tensor):
    """
    Fused 1x1 conv + reshape, working directly with NCHW format.
    """
    B, C, H, W = input_tensor.shape
    N = weight.shape[0]  # 17
    HW = H * W
    total_hw = B * HW
    
    # Flatten input to [B*C*HW] for easier indexing
    input_flat = input_tensor.view(-1)
    
    # Reshape weight to [N, C]
    weight_2d = weight.view(N, C)
    
    # Allocate output [B*HW, N]
    output = torch.empty((total_hw, N), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    grid = lambda META: (triton.cdiv(total_hw, META['BLOCK_SIZE']),)
    
    conv1x1_nchw_kernel[grid](
        input_flat, weight_2d, bias, output,
        B, C, HW, N,
    )
    
    # Reshape to final format [B, N, HW]
    output = output.view(B, HW, N).transpose(1, 2)
    
    return output

def replacement_func():
    return fused_conv1x1_mul_reshape