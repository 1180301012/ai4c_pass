import torch
import triton
import triton.language as tl

@triton.jit
def concat_dim2_kernel(
    in3_ptr,
    in4_ptr,
    tmp3_ptr,
    out_ptr,
    N, C, L3, L4, L_tmp3, L_total,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_l = tl.program_id(2)
    
    # Calculate output coordinates
    out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    out_l = pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    
    # Combined indices for output tensor
    output_indices = (out_n[:, None, None] * (C * L_total) + 
                     out_c[None, :, None] * L_total + 
                     out_l[None, None, :]).to(tl.int64)
    
    # Flatten indices for processing
    linear_indices = output_indices.flatten()
    
    # Process each output element
    for i, idx in enumerate(linear_indices):
        if idx < N * C * L_total:
            # Calculate relative positions
            rel_n = (idx // (C * L_total)) % N
            rel_c = (idx // L_total) % C
            rel_l = idx % L_total
            
            # Determine which input to load from
            if rel_l < L3:
                # Load from in3 (first part)
                input_idx = rel_n * (C * L3) + rel_c * L3 + rel_l
                val = tl.load(in3_ptr + input_idx)
            elif rel_l < L3 + L4:
                # Load from in4 (second part)
                rel_l_in4 = rel_l - L3
                input_idx = rel_n * (C * L4) + rel_c * L4 + rel_l_in4
                val = tl.load(in4_ptr + input_idx)
            else:
                # Load from tmp3 (third part)
                rel_l_in_tmp3 = rel_l - L3 - L4
                input_idx = rel_n * (C * L_tmp3) + rel_c * L_tmp3 + rel_l_in_tmp3
                val = tl.load(tmp3_ptr + input_idx)
            
            # Store to output
            tl.store(out_ptr + idx, val)

@torch.fx.wrap
def concat_dim2_optimized(in3, in4, tmp3):
    # Extract input dimensions
    N, C, L3 = in3.shape
    _, _, L4 = in4.shape
    _, _, L_tmp3 = tmp3.shape
    
    # Total length along dimension 2
    L_total = L3 + L4 + L_tmp3
    
    # Set grid dimensions
    grid = (
        (N + 7) // 8,    # N dimension blocks
        (C + 15) // 16,  # C dimension blocks
        (L_total + 15) // 16,  # L dimension blocks
    )
    
    # Create output tensor
    output = torch.empty((N, C, L_total), dtype=torch.float32, device="cuda")
    
    # Launch kernel
    concat_dim2_kernel[grid](
        in3,
        in4,
        tmp3,
        output,
        N, C, L3, L4, L_tmp3, L_total,
    )
    
    return output

def concat_pattern(input_tensor1, input_tensor2, input_tensor3):
    # Concatenation operation along dimension 2
    return torch.cat([input_tensor1, input_tensor2, input_tensor3], 2)

def replacement_args(in_3, in_4, tmp_3):
    return (in_3, in_4, tmp_3)

def replacement_func():
    return concat_dim2_optimized