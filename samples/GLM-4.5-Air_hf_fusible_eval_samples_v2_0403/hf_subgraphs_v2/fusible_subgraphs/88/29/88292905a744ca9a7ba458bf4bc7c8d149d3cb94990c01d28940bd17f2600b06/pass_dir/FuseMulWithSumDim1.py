import torch
import triton
import triton.language as tl

def pattern(tmp_3, in_0):
    tmp_4 = tmp_3 * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    return tmp_5

def replacement_args(tmp_3, in_0):
    return (tmp_3, in_0)

@triton.jit
def fused_mul_sum_kernel(
    tmp_3_ptr,
    in_0_ptr, 
    out_ptr,
    N_batch: tl.constexpr,
    N_channels_tmp3: tl.constexpr,
    N_spatial_dim1: tl.constexpr,
    N_spatial_dim2: tl.constexpr,
    N_spatial_dim3: tl.constexpr,
    N_channels_in0: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one spatial position in the output
    pid = tl.program_id(0)
    
    # Total elements in tmp_3: [N_batch, N_channels_tmp3, N_spatial_dim1, N_spatial_dim2, N_spatial_dim3]
    tmp_3_total_elements = N_channels_tmp3 * N_spatial_dim1 * N_spatial_dim2 * N_spatial_dim3
    
    # Total elements in sum result after reduction: [N_batch, N_spatial_dim1, N_spatial_dim2, N_spatial_dim3]
    output_total_elements = N_spatial_dim1 * N_spatial_dim2 * N_spatial_dim3
    
    if pid >= output_total_elements:
        return
    
    # Process this spatial position across all batch elements
    for batch_idx in range(N_batch):
        # For this batch and spatial position, sum across channels of (tmp_3 * in_0)
        sum_val = 0.0
        
        # Load tmp_3 data at this spatial position [N_channels_tmp3, 1, 1, 1] (due to view ops)
        tmp_3_base = batch_idx * tmp_3_total_elements + pid
        tmp_3_val = tl.load(tmp_3_ptr + tmp_3_base, mask=tmp_3_base < N_batch * tmp_3_total_elements)
        
        # Load in_0 data - need to handle broadcasting
        # in_0 shape: [N_batch, N_channels_in0, N_spatial_dim1, N_spatial_dim2, N_spatial_dim3] 
        # tmp_3 shape: [N_batch, N_channels_tmp3, 1, 1, 1] (from view ops)
        # Broadcasting: tmp_3 shape expands to [N_batch, N_channels_tmp3, N_spatial_dim1, N_spatial_dim2, N_spatial_dim3]
        
        # For each channel in tmp_3, multiply with corresponding in_0 element and sum
        # N_channels_tmp3 should equal N_channels_in0 due to broadcasting compatibility
        
        for ch_idx in range(N_channels_tmp3):
            in_0_base = batch_idx * N_channels_in0 * output_total_elements + ch_idx * output_total_elements + pid
            
            in_0_val = tl.load(in_0_ptr + in_0_base, mask=in_0_base < N_batch * N_channels_in0 * output_total_elements)
            
            # Multiply and accumulate
            sum_val += tmp_3_val * in_0_val
        
        # Store the summed result
        out_base = batch_idx * output_total_elements + pid
        tl.store(out_ptr + out_base, sum_val)

@torch.fx.wrap
def fused_mul_sum(tmp_3, in_0):
    N_batch = in_0.shape[0]
    N_channels_in0 = in_0.shape[1]
    N_spatial_dim1 = in_0.shape[2] 
    N_spatial_dim2 = in_0.shape[3]
    
    # Handle potential 5D tensor
    N_spatial_dim3 = 1
    if len(in_0.shape) > 4:
        N_spatial_dim3 = in_0.shape[4]
    
    # tmp_3 should be [N_batch, N_channels_tmp3, 1, 1, 1] after view ops
    tmp_3_expected_shape = [N_batch, N_channels_in0, 1, 1, 1]
    if tmp_3.shape != tuple(tmp_3_expected_shape):
        # If it doesn't match, reshape it to match our expected pattern
        if len(tmp_3.shape) == 2 and tmp_3.shape[0] == N_batch:
            # Case: [N_batch, N_channels_tmp3_total] where N_channels_tmp3_total should be N_channels_in0
            N_channels_tmp3_total = tmp_3.shape[1]
            if N_channels_tmp3_total == N_channels_in0:
                tmp_3 = tmp_3.view(N_batch, N_channels_in0, 1, 1, 1)
            else:
                raise ValueError(f"Shape mismatch: expected {tmp_3_expected_shape}, got {tmp_3.shape}")
        else:
            raise ValueError(f"Shape mismatch: expected {tmp_3_expected_shape}, got {tmp_3.shape}")
    
    # Output should be same as in_0 but with channels summed: [N_batch, N_spatial_dim1, N_spatial_dim2, N_spatial_dim3]
    if len(in_0.shape) == 5:
        output_shape = [N_batch, N_spatial_dim1, N_spatial_dim2, N_spatial_dim3]
    else:
        output_shape = [N_batch, N_spatial_dim1, N_spatial_dim2]
    
    out = torch.empty(output_shape, dtype=tmp_3.dtype, device=tmp_3.device)
    
    # Launch kernel
    output_total_elements = N_spatial_dim1 * N_spatial_dim2 * N_spatial_dim3
    BLOCK_SIZE = 1024
    fused_mul_sum_kernel[(output_total_elements,)](
        tmp_3_ptr=tmp_3,
        in_0_ptr=in_0,
        out_ptr=out,
        N_batch=N_batch,
        N_channels_tmp3=N_channels_in0,
        N_spatial_dim1=N_spatial_dim1,
        N_spatial_dim2=N_spatial_dim2,
        N_spatial_dim3=N_spatial_dim3,
        N_channels_in0=N_channels_in0,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_mul_sum