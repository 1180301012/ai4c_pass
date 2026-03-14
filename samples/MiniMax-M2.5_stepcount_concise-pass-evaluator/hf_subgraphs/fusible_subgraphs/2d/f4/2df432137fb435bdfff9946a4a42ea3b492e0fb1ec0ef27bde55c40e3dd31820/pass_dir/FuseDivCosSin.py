import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Match the computation pattern:
    - Creates meshgrid from in_1 and arange
    - Flattens both meshgrid outputs
    - Reshapes in_0 to (1, -1)
    - Divides flattened tensors by reshaped in_0 (broadcasting)
    - Applies cos to first division result
    - Applies sin to first division result
    - Returns (cos(div1), div2, sin(div1))
    """
    # Create arange - use 'cuda' as device to match model
    tmp_1 = torch.arange(8, dtype=torch.float32, device='cuda')
    
    # Meshgrid
    tmp_2 = torch.functional.meshgrid(in_1, tmp_1)
    
    # Extract meshgrid outputs
    tmp_3 = tmp_2[0]
    tmp_4 = tmp_2[1]
    
    # Flatten
    tmp_5 = tmp_3.flatten()
    tmp_6 = tmp_4.flatten()
    
    # Reshape in_0
    tmp_7 = in_0.reshape(1, -1)
    
    # First division path (for cos and sin)
    tmp_8 = tmp_5.unsqueeze(-1)
    tmp_9 = tmp_8 / tmp_7
    
    # Second division path
    tmp_10 = tmp_6.unsqueeze(-1)
    tmp_11 = tmp_10 / tmp_7
    
    # Apply cos and sin to first division result
    tmp_12 = tmp_9.cos()
    tmp_13 = tmp_9.sin()
    
    return (tmp_12, tmp_11, tmp_13)


def replacement_args(in_0, in_1):
    """
    Extract arguments needed for the replacement kernel.
    """
    return (in_0, in_1)


@triton.jit
def fused_div_cos_sin_kernel(
    in_0_ptr, in_1_ptr,
    out_cos_ptr, out_div2_ptr, out_sin_ptr,
    n_in_0, n_in_1,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes:
    - meshgrid + flatten + divide + cos for first path
    - meshgrid + flatten + divide for second path
    """
    # Grid: each program handles one element of the output
    # Output is n_in_0 * n_in_1 (64 * 8 = 512, since meshgrid creates 8x8)
    # Actually we need to generate n_in_1 * n_in_1 elements (8 * 8 = 64)
    program_id = tl.program_id(0)
    
    # Compute row and col indices (meshgrid indices)
    # The computation creates n_in_1 * n_in_1 elements
    row_idx = program_id // n_in_1
    col_idx = program_id % n_in_1
    
    # Load the in_1 value (row index from input)
    # But wait, the meshgrid is created from in_1 and arange
    # tmp_3 = tmp_2[0] uses in_1 as first dim, arange as second
    # tmp_4 = tmp_2[1] uses in_1 as first dim, arange as second
    
    # Actually let's think again:
    # meshgrid(in_1, tmp_1) where in_1 is [8] and tmp_1 is [8]
    # Result: tuple of two [8, 8] tensors
    # tmp_3 = tmp_2[0] is row tensor (varies along axis 1)
    # tmp_4 = tmp_2[1] is col tensor (varies along axis 0)
    
    # So for flattened:
    # tmp_5 comes from tmp_3.flatten() - row-major order
    # tmp_6 comes from tmp_4.flatten() - column-major order
    
    # Load in_1[row_idx] for tmp_3 (first meshgrid output)
    in_1_idx = row_idx
    val_tmp3 = tl.load(in_1_ptr + in_1_idx)
    
    # For tmp_4 (second meshgrid output), it's arange values
    val_tmp4 = col_idx.to(tl.float32)
    
    # Now compute division by in_0 elements
    # We need to divide by each element of in_0
    # The result is stored at out[row_idx * n_in_0 + col_idx], but actually...
    # tmp_9 = tmp_8 / tmp_7 where tmp_8 is [64, 1] and tmp_7 is [1, 64]
    # Result is [64, 64] where element [i, j] = tmp_5[i] / tmp_7[j]
    
    # For the output, we need:
    # tmp_12[i, j] = cos(tmp_5[i] / in_0[j])
    # tmp_11[i, j] = tmp_6[i] / in_0[j]
    # tmp_13[i, j] = sin(tmp_5[i] / in_0[j])
    
    # The program_id maps to (i, j) pair
    i = program_id // n_in_0  # This is wrong, need to reconsider
    j = program_id % n_in_0
    
    # Actually let's re-analyze the shapes:
    # in_0 has shape [64]
    # in_1 has shape [8]
    # meshgrid produces [8, 8] tensors
    # flatten gives [64]
    # unsqueeze gives [64, 1]
    # reshape in_0 gives [1, 64]
    # division gives [64, 64]
    
    # So the output has shape [64, 64] = 4096 elements
    # But wait, n_in_1 = 8, so 8 * 8 = 64... but output is 64 * 64 = 4096
    
    # Let me reconsider: tmp_5 and tmp_6 each have 64 elements (from 8x8 meshgrid)
    # tmp_7 has 64 elements (from in_0 reshape)
    # So division creates 64 * 64 = 4096 elements
    
    # For program_id, we need:
    i = program_id // n_in_0  # 0-63 (from flattened meshgrid index)
    j = program_id % n_in_0   # 0-63 (from in_0 index)
    
    # But wait, flattened meshgrid has 64 elements (8x8)
    # in_0 has 64 elements
    # So output is 64x64
    
    # For tmp_3 = meshgrid(in_1, arange)[0], at flattened index i:
    # i = row * 8 + col, row from 0-7, col from 0-7
    # tmp_3[row, col] = in_1[row]
    row_idx = i // n_in_1
    col_idx = i % n_in_1
    
    # tmp_3 value at (row_idx, col_idx) = in_1[row_idx]
    val_tmp3 = tl.load(in_1_ptr + row_idx)
    
    # tmp_4 value at (row_idx, col_idx) = arange[col_idx] = col_idx
    val_tmp4 = col_idx.to(tl.float32)
    
    # Division by in_0[j]
    div1 = val_tmp3 / tl.load(in_0_ptr + j)
    div2 = val_tmp4 / tl.load(in_0_ptr + j)
    
    # Compute cos and sin
    cos_result = tl.cos(div1)
    sin_result = tl.sin(div1)
    
    # Store results
    tl.store(out_cos_ptr + program_id, cos_result)
    tl.store(out_div2_ptr + program_id, div2)
    tl.store(out_sin_ptr + program_id, sin_result)


@torch.fx.wrap
def fused_div_cos_sin(in_0, in_1):
    """
    Fused kernel that computes the sinusoidal positional encoding:
    - Computes meshgrid, flatten, divide, cos/sin in a single kernel
    """
    n_in_0 = in_0.shape[0]  # 64
    n_in_1 = in_1.shape[0]  # 8
    
    # Output shape is [64, 64] = n_in_0 * n_in_0
    # Since meshgrid creates n_in_1 * n_in_1 = 64 elements
    # and we divide by n_in_0 = 64 elements
    output_size = n_in_0 * n_in_0  # 4096
    
    # Allocate output tensors - use 'cuda' device directly for symbolic tracing
    out_cos = torch.empty((n_in_0, n_in_0), dtype=torch.float32, device='cuda')
    out_div2 = torch.empty((n_in_0, n_in_0), dtype=torch.float32, device='cuda')
    out_sin = torch.empty((n_in_0, n_in_0), dtype=torch.float32, device='cuda')
    
    # Launch kernel
    BLOCK_SIZE = 64
    num_programs = output_size
    
    fused_div_cos_sin_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_cos_ptr=out_cos,
        out_div2_ptr=out_div2,
        out_sin_ptr=out_sin,
        n_in_0=n_in_0,
        n_in_1=n_in_1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (out_cos, out_div2, out_sin)


def replacement_func():
    return fused_div_cos_sin