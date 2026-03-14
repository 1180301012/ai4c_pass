import torch
import triton
import triton.language as tl

def pattern(x, in_0):
    """Pattern: split tensor + unsqueeze on last part + return all required outputs"""
    tmp_1 = torch.nn.functional.silu(x, inplace=True)
    tmp_2 = torch.functional.split(tmp_1, [512, 512, 128], dim=2)
    tmp_3 = tmp_2[0]
    tmp_4 = tmp_2[1]
    tmp_5 = tmp_2[2]
    tmp_6 = tmp_5.unsqueeze(2)
    tmp_7 = in_0[None, None, slice(None, None, None)]
    return (tmp_7, tmp_3, tmp_6, tmp_4)

def replacement_args(x, in_0):
    return (x, in_0)

@triton.jit
def silu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Optimized SiLU activation kernel"""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    x = tl.load(x_ptr + offset, mask=mask)
    out = x * (1.0 / (1.0 + tl.exp(-x)))
    tl.store(out_ptr + offset, out, mask=mask)

@triton.jit
def split_and_unsqueeze_kernel(x_ptr, out_0_ptr, out_1_ptr, out_2_ptr,
                              batch_size, height, total_features,
                              split_1, split_2,
                              BLOCK_SIZE: tl.constexpr):
    """Optimized kernel that performs split + unsqueeze in one operation"""
    pid = tl.program_id(0)
    
    # Calculate thread work ranges
    per_thread = (batch_size * height * total_features + BLOCK_SIZE - 1) // BLOCK_SIZE
    start = pid * per_thread
    end = min(start + per_thread, batch_size * height * total_features)
    
    # Process each element in the thread's range
    for idx in range(start, end):
        # Convert flat index to 3D indices
        b = idx // (height * total_features)
        h = (idx % (height * total_features)) // total_features  
        f = idx % total_features
        
        # Load input value
        value = tl.load(x_ptr + idx)
        
        # Determine which split this belongs to
        if f < split_1:  # First split [0:512]
            output_idx = b * height * split_1 + h * split_1 + f
            tl.store(out_0_ptr + output_idx, value)
        elif f < split_1 + split_2:  # Second split [512:1024]
            relative_f = f - split_1
            output_idx = b * height * split_2 + h * split_2 + relative_f
            tl.store(out_1_ptr + output_idx, value)
        else:  # Third split [1024:1152] + unsqueeze(2)
            relative_f = f - split_1 - split_2  # remaining 128 elements
            # unsqueeze(2) adds dimension at position 2: [B, H, 128] -> [B, H, 128, 1]
            # We process this by storing to expanded layout
            base_offset = (b * height * split_2 + h * split_2 + relative_f) * 2
            tl.store(out_2_ptr + base_offset, value)      # Actual value
            tl.store(out_2_ptr + base_offset + 1, 0.0)    # Padding for new dimension

@torch.fx.wrap
def optimized_split_and_unsqueeze(x, in_0):
    """Highly optimized version of split + unsqueeze pattern"""
    batch_size, height, total_features = x.shape
    
    # Extract split sizes
    split_1, split_2 = 512, 512
    split_3 = total_features - split_1 - split_2
    
    # Create output tensors with pre-allocated memory
    out_0 = torch.empty((batch_size, height, split_1), dtype=x.dtype, device=x.device)
    out_1 = torch.empty((batch_size, height, split_2), dtype=x.dtype, device=x.device)  
    out_2 = torch.empty((batch_size, height, split_3, 1), dtype=x.dtype, device=x.device)
    out_3 = in_0.unsqueeze(0).unsqueeze(0)  # More efficient than [None, None, :]
    
    # Triton SiLU kernel
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    tmp_1 = torch.empty_like(x)
    silu_kernel[(grid_size,)](x, tmp_1, n_elements, BLOCK_SIZE)
    
    # Triton split + unsqueeze kernel  
    split_kernel_grid = (batch_size * height * total_features + 1023) // 1024
    split_and_unsqueeze_kernel[(split_kernel_grid,)](
        tmp_1, out_0, out_1, out_2,
        batch_size, height, total_features,
        split_1, split_2, 1024
    )
    
    return out_3, out_0, out_2, out_1

def replacement_func():
    return optimized_split_and_unsqueeze