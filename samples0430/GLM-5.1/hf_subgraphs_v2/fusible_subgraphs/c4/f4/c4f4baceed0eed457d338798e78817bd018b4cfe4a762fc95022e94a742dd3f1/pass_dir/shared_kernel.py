import triton
import triton.language as tl


@triton.jit
def fused_roll_slice_add_layernorm_kernel(
    in_3_ptr, in_2_ptr, weight_ptr, bias_ptr,
    out_add_ptr, out_ln_ptr,
    N, C, H_4d, W_4d, H_cut, W_cut, shift,
    eps,
    BLOCK_C: tl.constexpr,
):
    """
    Fused kernel that computes:
    1. Roll + slice on in_3 (spatial shift with cyclic padding)
    2. Add with in_2 (residual addition)
    3. Layer norm with weight and bias
    
    Each program processes one row (all C elements) of the output.
    """
    row_idx = tl.program_id(0)
    if row_idx >= N:
        return
    
    # Compute spatial position in the output grid
    i = row_idx // W_cut  # row in 2D spatial grid
    j = row_idx % W_cut   # col in 2D spatial grid
    
    # Compute source position after roll (cyclic shift)
    # torch.roll(x, shifts=shift, dims=1) means: result[j] = x[(j - shift) % size]
    source_h = (i - shift) % H_4d
    source_w = (j - shift) % W_4d
    
    # Channel offsets
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    
    # Load from in_3 at the source position (4D layout: [1, H_4d, W_4d, C])
    in_3_offset = (source_h * W_4d + source_w) * C + offs_c
    in_3_vals = tl.load(in_3_ptr + in_3_offset, mask=mask_c, other=0.0).to(tl.float32)
    
    # Load from in_2 (3D layout: [1, N, C], row_idx corresponds to spatial position)
    in_2_offset = row_idx * C + offs_c
    in_2_vals = tl.load(in_2_ptr + in_2_offset, mask=mask_c, other=0.0).to(tl.float32)
    
    # Addition (tmp_8 = in_2 + rolled_in_3)
    add_vals = in_3_vals + in_2_vals
    
    # Layer Norm computation in float32 for precision
    # Step 1: Compute mean
    # Masked positions are 0.0, so sum of all elements = sum of valid elements
    mean = tl.sum(add_vals, axis=0) / C
    
    # Step 2: Compute variance (need to mask out invalid positions)
    diff = add_vals - mean
    diff_sq = tl.where(mask_c, diff * diff, 0.0)
    var = tl.sum(diff_sq, axis=0) / C
    
    # Step 3: Normalize
    rstd = 1.0 / tl.sqrt(var + eps)
    normalized = tl.where(mask_c, diff * rstd, 0.0)
    
    # Step 4: Apply weight and bias
    weight_vals = tl.load(weight_ptr + offs_c, mask=mask_c, other=1.0).to(tl.float32)
    bias_vals = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    ln_vals = normalized * weight_vals + bias_vals
    
    # Store addition output (tmp_8)
    tl.store(out_add_ptr + row_idx * C + offs_c, add_vals, mask=mask_c)
    
    # Store layer norm output (tmp_9)
    tl.store(out_ln_ptr + row_idx * C + offs_c, ln_vals, mask=mask_c)