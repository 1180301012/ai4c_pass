import torch
import triton
import triton.language as tl

def pattern(x, attention_mask):
    # Step 1: Create mask where attention is allowed (to use attention_mask parameter)
    mask = attention_mask.__eq__(0)
    
    # Step 2: Cumulative sum along last dimension
    cumsum = x.cumsum(-1)
    
    # Step 3: Subtract 1
    cumsum_minus_1 = cumsum - 1
    
    # Step 4: Apply masked fill (as function call, not method call)
    result = torch.ops.aten.masked_fill_(cumsum_minus_1, mask, 1)
    
    # Step 5: Unsqueeze and expand for 3 parallel operations
    unsqueezed = result.unsqueeze(0)
    expanded = unsqueezed.expand(3, -1, -1)
    
    return expanded

def replacement_args(x, attention_mask):
    return (x, attention_mask)

@triton.jit
def fused_cumsum_expand_kernel(
    x_ptr,
    mask_ptr,
    out_ptr,
    batch_size,
    seq_len,
    num_expanded,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one element in the output
    m = tl.program_id(0)
    n = tl.program_id(1)
    k = tl.program_id(2)
    
    # Define range of columns this program handles
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < seq_len
    
    # Create offset for batch/expanded dimensions
    row_offset = (m * num_expanded + k) * seq_len
    col_offset = cols
    
    # Calculate indices
    indices = row_offset + col_offset
    
    if row_offset >= batch_size * num_expanded * seq_len:
        return
    
    # Load x and mask values for this batch position
    if m < batch_size:
        # Base index for this batch
        base_idx = m * seq_len
        
        # Compute cumulative sum for this row
        cumsum_val = 0
        for i in range(seq_len):
            if i < seq_len:
                curr_idx = base_idx + i
                curr_val = tl.load(x_ptr + curr_idx, mask=i < seq_len, other=0)
                cumsum_val = cumsum_val + curr_val
        
        # Apply cumsum - 1
        cumsum_minus_1 = cumsum_val - 1
        
        # Load mask values for this batch and apply masking
        # We'll create a simple masked fill by checking if any mask value is 0
        # and if so, use 1, otherwise use cumsum_minus_1
        mask_vals = tl.load(mask_ptr + base_idx, mask=tl.arange(0, seq_len) < seq_len, other=1)
        has_attention_allowed = tl.any(mask_vals == 0)
        
        # Apply masked fill: if attention is allowed anywhere, use cumsum_minus_1, else use 1
        fill_val = tl.where(has_attention_allowed, cumsum_minus_1, 1)
        
        out_vals = fill_val
    else:
        # Fill with zeros for invalid batch positions
        out_vals = tl.zeros((BLOCK_SIZE_N,), dtype=tl.int64)
    
    # Store results
    tl.store(out_ptr + indices, out_vals, mask=mask)

@torch.fx.wrap
def fused_cumsum_mask_expand(x, attention_mask):
    batch_size, seq_len = x.shape
    num_expanded = 3  # Expand to 3 parallel operations
    
    # Create output shape: (num_expanded, batch_size, seq_len)
    expanded_shape = (num_expanded, batch_size, seq_len)
    out = torch.empty(expanded_shape, dtype=torch.int64, device=x.device)
    
    # Determine grid size
    BLOCK_SIZE_M = 1
    BLOCK_SIZE_N = 1024  # Tune based on sequence length
    
    num_blocks_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    fused_cumsum_expand_kernel[
        (num_blocks_m, num_blocks_n, num_expanded)
    ](
        x_ptr=x,
        mask_ptr=attention_mask,
        out_ptr=out.view(-1),  # Flatten for kernel
        batch_size=batch_size,
        seq_len=seq_len,
        num_expanded=num_expanded,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    # Ensure output is on CUDA device (handles the .to(device()) call)
    if out.device.type != 'cuda':
        out = out.to(device='cuda:0')
    
    return out

def replacement_func():
    return fused_cumsum_mask_expand