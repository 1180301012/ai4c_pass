import torch
import triton
import triton.language as tl

def pattern(input_1, input_0):
    # Match the complete sequence: softmax -> reshape -> view -> view -> multiply -> sum
    tmp_0 = torch.nn.functional.softmax(input_1, dim=1)
    tmp_1 = tmp_0.reshape(-1)
    tmp_2 = tmp_1.view(-1, 1, 1)
    tmp_3 = tmp_2.view(2, -1, 1, 1)
    tmp_4 = tmp_3 * input_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    return tmp_5

def replacement_args(input_1, input_0):
    return (input_1, input_0)

@triton.jit
def fused_softmax_multiply_sum_kernel(
    input_1_ptr,
    input_0_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    two_dim: tl.constexpr,
    one_dim: tl.constexpr,
    hidden_dim: tl.constexpr,
    spatial_h: tl.constexpr,
    spatial_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    spatial_elements = spatial_h * spatial_w
    offset = pid * BLOCK_SIZE
    mask = offset + tl.arange(0, BLOCK_SIZE) < spatial_elements
    
    # Map spatial index to h, w coordinates
    spatial_idx = offset + tl.arange(0, BLOCK_SIZE)
    h = spatial_idx // spatial_w
    w = spatial_idx % spatial_w
    
    # Process one batch element at a time
    batch_idx = pid // spatial_elements
    mask_batch = batch_idx < batch_size
    
    result = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
    
    # For each hidden dimension, compute softmax and multiply
    for k in range(0, hidden_dim, 1):
        hidden_idx = k
        
        # Load input_1 segment for softmax
        input_1_start = (batch_idx * (two_dim * one_dim * hidden_dim) + 
                        (k % two_dim) * (one_dim * hidden_dim) + 
                        hidden_idx)
        
        # Load input_1 values for this hidden dimension
        input_1_values = tl.load(input_1_ptr + input_1_start + tl.arange(0, two_dim * one_dim), 
                                mask=tl.arange(0, two_dim * one_dim) < (two_dim * one_dim), other=0.0)
        
        # Compute softmax along the two_dim (dim=1)
        max_val = tl.max(input_1_values)
        exp_vals = tl.exp(input_1_values - max_val)
        sum_exp = tl.sum(exp_vals)
        softmax_vals = exp_vals / sum_exp
        
        # For each channel dimension
        for c in range(0, two_dim):
            channel_idx = c
            
            # Broadcast softmax value across spatial dimensions
            softmax_val = softmax_vals[c]
            
            # Compute indices for input_0 (which should have shape [batch, 2, hidden_dim, spatial_h, spatial_w])
            input_0_start = (batch_idx * (two_dim * hidden_dim * spatial_h * spatial_w) + 
                           channel_idx * (hidden_dim * spatial_h * spatial_w) +
                           hidden_idx * (spatial_h * spatial_w) +
                           h * spatial_w + w)
            
            # Load input_0 values with masking
            input_0_vals = tl.load(input_0_ptr + input_0_start, 
                                  mask=mask & (h < spatial_h) & (w < spatial_w), 
                                  other=0.0)
            
            # Multiply and accumulate (sum over channel dimension)
            result += softmax_val * input_0_vals
    
    # Store result for this spatial location
    out_idx = batch_idx * (spatial_h * spatial_w) + spatial_idx
    tl.store(out_ptr + out_idx, result, mask=mask & mask_batch)

@torch.fx.wrap
def fused_softmax_multiply_sum(input_1, input_0):
    # Get input shapes
    batch_size, two_dim, one_dim, hidden_dim = input_1.shape
    spatial_h = input_0.shape[3]
    spatial_w = input_0.shape[4]
    
    # Output shape after sum over dim=1 (two_dim)
    out_shape = (batch_size, hidden_dim, spatial_h, spatial_w)
    out = torch.zeros(out_shape, dtype=input_1.dtype, device=input_1.device)
    
    # Configure kernel launch
    total_spatial_elements = batch_size * spatial_h * spatial_w
    BLOCK_SIZE = 1024
    num_programs = (total_spatial_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_softmax_multiply_sum_kernel[(num_programs,)](
        input_1_ptr=input_1,
        input_0_ptr=input_0,
        out_ptr=out,
        batch_size=batch_size,
        two_dim=two_dim,
        one_dim=one_dim,
        hidden_dim=hidden_dim,
        spatial_h=spatial_h,
        spatial_w=spatial_w,
        BLOCK_SIZE=BLOCK_SIZE,
        DIM=1,  # The dimension for softmax (dim=1)
    )
    
    return out

def replacement_func():
    return fused_softmax_multiply_sum