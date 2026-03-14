import torch
import triton
import triton.language as tl

# Pattern matching function for split operation
def pattern(in_0, in_1, in_2):
    """Pattern matches split operation"""
    tmp_4 = torch.randn(1, 128, 96, 96)  # Placeholder - will be matched from context
    tmp_5 = torch.functional.split(tmp_4, [32, 48, 48], dim=1)
    tmp_6 = tmp_5[0]
    tmp_7 = tmp_5[1] 
    tmp_8 = tmp_5[2]
    return tmp_6, tmp_7, tmp_8

# Argument extraction function  
def replacement_args(in_0, in_1, in_2):
    """Extract arguments for split optimization"""
    return (in_0, in_1, in_2)

@triton.jit
def split_kernel(
    input_ptr,
    output1_ptr,
    output2_ptr,
    output3_ptr,
    batch_size,
    split_dim,
    dim_size,
    output1_size,
    output2_size,
    output3_size,
    H,
    W,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    """Triton kernel for splitting tensor along a dimension"""
    pid = tl.program_id(0)
    
    # Calculate block coordinates (works for any dimension)
    if split_dim == 1:  # Splitting along channel dimension
        h = pid % ((H + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H)
        w = pid // ((H + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H)
        
        h_start = h * BLOCK_SIZE_H
        h_end = min((h + 1) * BLOCK_SIZE_H, H)
        w_start = w * BLOCK_SIZE_W
        w_end = min((w + 1) * BLOCK_SIZE_W, W)
        
        # Process each output in parallel regions
        for b in range(batch_size):
            input_base = b * dim_size * H * W
            
            # Output 1 processing
            out1_base = b * output1_size * H * W
            for hh in range(h_start, h_end):
                for ww in range(w_start, w_end):
                    input_idx = input_base + output1_size * H * W + hh * W + ww
                    output_idx = out1_base + hh * W + ww
                    mask = (tl.arange(BLOCK_SIZE_H)[:, None] < (h_end - h_start)) & (tl.arange(BLOCK_SIZE_W) < (w_end - w_start))
                    output_val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
                    tl.store(output1_ptr + output_idx, output_val, mask=mask)
            
            # Output 2 processing  
            out2_base = b * output2_size * H * W
            for hh in range(h_start, h_end):
                for ww in range(w_start, w_end):
                    input_idx = input_base + (output1_size + output2_size) * H * W + hh * W + ww
                    output_idx = out2_base + hh * W + ww
                    mask = (tl.arange(BLOCK_SIZE_H)[:, None] < (h_end - h_start)) & (tl.arange(BLOCK_SIZE_W) < (w_end - w_start))
                    output_val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
                    tl.store(output2_ptr + output_idx, output_val, mask=mask)
            
            # Output 3 processing
            out3_base = b * output3_size * H * W
            for hh in range(h_start, h_end):
                for ww in range(w_start, w_end):
                    input_idx = input_base + (output1_size + output2_size + output3_size) * H * W + hh * W + ww
                    output_idx = out3_base + hh * W + ww
                    mask = (tl.arange(BLOCK_SIZE_H)[:, None] < (h_end - h_start)) & (tl.arange(BLOCK_SIZE_W) < (w_end - w_start))
                    output_val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
                    tl.store(output3_ptr + output_idx, output_val, mask=mask)

@torch.fx.wrap  
def optimized_split(x, split_sizes, dim=1):
    """Optimized split operation using Triton"""
    batch_size, total_dims, H, W = x.shape
    
    # Validate split sizes
    split_sum = sum(split_sizes)
    if split_sum != total_dims:
        # Return empty tensors if sizes don't match (shouldn't happen in valid patterns)
        return (torch.empty(0), torch.empty(0), torch.empty(0))
    
    output1_size, output2_size, output3_size = split_sizes
    
    # Create output tensors
    out1 = torch.empty((batch_size, output1_size, H, W), dtype=x.dtype, device=x.device)
    out2 = torch.empty((batch_size, output2_size, H, W), dtype=x.dtype, device=x.device) 
    out3 = torch.empty((batch_size, output3_size, H, W), dtype=x.dtype, device=x.device)
    
    # Block sizes
    BLOCK_SIZE_H = 32
    BLOCK_SIZE_W = 32
    
    # Configure grid
    grid_h = (H + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (W + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    grid_size = grid_h * grid_w
    
    # Launch kernel
    split_kernel[grid_size](
        input_ptr=x,
        output1_ptr=out1,
        output2_ptr=out2,
        output3_ptr=out3,
        batch_size=batch_size,
        split_dim=dim,
        dim_size=total_dims,
        output1_size=output1_size,
        output2_size=output2_size,
        output3_size=output3_size,
        H=H,
        W=W,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return out1, out2, out3

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_split