import torch
import triton
import triton.language as tl

# Pattern matching function for the entire forward pass
def pattern(in_0, in_1):
    """Match both operations in sequence: normalization + view+expand"""
    tmp_0 = in_1.sum(dim=2, keepdim=True)
    tmp_1 = in_1 / tmp_0
    tmp_2 = in_0.view(1, 2, 1, 8, 8)
    tmp_3 = tmp_2.expand(1, 2, 64, 8, 8)
    return tmp_3, tmp_1


# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)


# Triton kernel for optimized dual operations
@triton.jit
def optimized_dual_operations_kernel(
    in0_ptr, in1_ptr, 
    out0_ptr, out1_ptr,
    n_batch,
    n_channels,
    n_h,
    n_w,
):
    """Combined kernel for normalization and broadcasting"""
    batch = tl.program_id(0) // n_channels
    channel = tl.program_id(0) % n_channels
    
    # For each (h,w) position, process both operations
    for h_idx in range(n_h):
        for w_idx in range(n_w):
            # Process normalization operation on in1 -> output tmp_1
            sum_val = 0.0
            # Sum along dimension 2 for this (h,w) position
            for k in range(8):
                src_offset = (batch * (n_channels * 8 * n_h * n_w) + 
                            channel * (8 * n_h * n_w) + 
                            k * (n_h * n_w) + 
                            h_idx * n_w + w_idx)
                val = tl.load(in1_ptr + src_offset)
                sum_val = sum_val + val.to(tl.float32)
            
            # Compute inverse sum
            if sum_val != 0:
                inv_sum = 1.0 / sum_val
            else:
                inv_sum = 0.0
            
            # Apply normalization for all k in dimension 2
            for k in range(8):
                src_offset = (batch * (n_channels * 8 * n_h * n_w) + 
                            channel * (8 * n_h * n_w) + 
                            k * (n_h * n_w) + 
                            h_idx * n_w + w_idx)
                dst_offset = (batch * (n_channels * 8 * n_h * n_w) + 
                            channel * (8 * n_h * n_w) + 
                            k * (n_h * n_w) + 
                            h_idx * n_w + w_idx)
                
                val = tl.load(in1_ptr + src_offset)
                result = val * inv_sum.to(val.dtype)
                tl.store(out1_ptr + dst_offset, result)
            
            # Process broadcasting operation on in0 -> output tmp_3 (with 64 expansion)
            # For the original tensor [1, 2, 8, 8] -> expanded [1, 2, 64, 8, 8]
            src_offset = (batch * (n_channels * n_h * n_w) + 
                        channel * (n_h * n_w) + 
                        h_idx * n_w + w_idx)
            
            # Broadcast to all 64 expansion dimensions
            for exp_dim in range(64):
                dst_offset = (batch * (n_channels * 64 * n_h * n_w) + 
                            channel * (64 * n_h * n_w) + 
                            exp_dim * (n_h * n_w) + 
                            h_idx * n_w + w_idx)
                
                src_val = tl.load(in0_ptr + src_offset)
                tl.store(out0_ptr + dst_offset, src_val)


@torch.fx.wrap
def optimized_dual_operations(in_0, in_1):
    """Optimized forward pass combining both operations"""
    n_batch, n_channels, n_h, n_w = in_0.shape[0], in_0.shape[1], in_0.shape[2], in_0.shape[3]
    
    # Create output tensors
    tmp_3 = torch.empty((n_batch, n_channels, 64, n_h, n_w), device=in_0.device, dtype=in_0.dtype)
    tmp_1 = torch.empty_like(in_1)
    
    # Launch optimized kernel
    grid_size = n_batch * n_channels
    optimized_dual_operations_kernel[(grid_size,)](
        in_0, in_1,
        tmp_3, tmp_1,
        n_batch, n_channels, n_h, n_w
    )
    
    return tmp_3, tmp_1


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_dual_operations