import torch
import triton
import triton.language as tl


@triton.jit
def fused_einsum_cat_softmax_kernel(
    # Pointers
    energy_ptr, query_ptr, key_ptr, softmax_out_ptr, sliced_out_ptr,
    # Shapes
    B: tl.constexpr, H: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
    # Strides
    energy_s0, energy_s1, energy_s2, energy_s3,
    query_s0, query_s1, query_s2, query_s3,
    key_s0, key_s1, key_s2, key_s3,
    out_s0, out_s1, out_s2, out_s3,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused einsum + cat + softmax kernel.
    
    einsum: 'bchw,bchj->bhwj' with query=B,C,H,W and key=B,C,H,J
    Result: [B, H, W, J]
    
    cat: concatenate [energy] + [einsum_result] along last dim
    Result: [B, H, W, J*2] where J = C = 64
    
    softmax: compute softmax over last dimension
    Result: [B, H, W, J*2]
    """
    # Get position
    pid = tl.program_id(0)
    
    # Calculate which block this program processes
    total_elements = B * H * W * C
    block_start = pid * BLOCK_SIZE
    block_end = block_start + BLOCK_SIZE
    
    # Create masks
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements - block_start
    
    # Calculate indices for this block
    base_idx = block_start + offsets
    
    # Compute multi-dimensional indices: [b, h, w, j]
    j = base_idx // (B * H * W)
    temp = base_idx % (B * H * W)
    w = temp // (B * H)
    temp = temp % (B * H)
    h = temp // B
    b = temp % B
    
    # Compute flat indices for loading data
    # energy[b, h, w, :] - need to access all J elements for softmax computation
    energy_idx = b * energy_s0 + h * energy_s1 + w * energy_s2 + j * energy_s3
    
    # query[b, c, h, w] and key[b, c, h, j]
    # For einsum 'bchw,bchj->bhwj': sum over c
    query_idx = b * query_s0 + h * query_s2 + w * query_s3 + tl.arange(0, BLOCK_SIZE) % C * query_s1
    
    # Load energy values
    energy_vals = tl.load(energy_ptr + energy_idx, mask=mask, other=0.0)
    
    # Compute einsum: query @ key over C dimension -> [b, h, w, j]
    # This is a batched dot product: sum_c query[b, c, h, w] * key[b, c, h, j]
    einsum_vals = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for c_idx in range(C):
        # query[b, c, h, w]
        q_idx = b * query_s0 + c_idx * query_s1 + h * query_s2 + w * query_s3
        q_val = tl.load(query_ptr + q_idx, mask=c_idx < C, other=0.0)
        
        # key[b, c, h, j]
        k_idx = b * key_s0 + c_idx * key_s1 + h * key_s2 + j * key_s3
        k_val = tl.load(key_ptr + k_idx, mask=(c_idx < C) & mask, other=0.0)
        
        einsum_vals += q_val * k_val
    
    # Concatenate: first half is energy, second half is einsum
    # For position j, if j < C use energy[j], else use einsum[j-C]
    # We'll store the concatenated value for softmax computation
    concat_val = tl.where(j < C, energy_vals, einsum_vals)
    
    # Store concatenated values for later use in denominator computation
    # We need to compute exp(x) and sum exp(x) for softmax
    concat_ptr = softmax_out_ptr + b * out_s0 + h * out_s1 + w * out_s2 + j * out_s3
    
    # Load full row for denominator computation
    # First, compute exp for this element
    exp_val = tl.exp(tl.cast(concat_val, tl.float32))
    tl.store(concat_ptr, exp_val, mask=mask)
    
    # For sliced output, we only need first half (j < C)
    sliced_ptr = sliced_out_ptr + b * out_s0 + h * out_s1 + w * out_s2 + j * out_s3
    sliced_val = tl.where(j < C, exp_val, 0.0)
    tl.store(sliced_ptr, sliced_val, mask=mask)


@triton.jit
def softmax_normalize_kernel(
    softmax_out_ptr, sliced_out_ptr,
    B: tl.constexpr, H: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
    out_s0, out_s1, out_s2, out_s3,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Second pass: compute sum of exponentials (denominator) and normalize.
    """
    pid = tl.program_id(0)
    total_elements = B * H * W * C
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements - block_start
    base_idx = block_start + offsets
    
    # Compute indices
    j = base_idx // (B * H * W)
    temp = base_idx % (B * H * W)
    w = temp // (B * H)
    temp = temp % (B * H)
    h = temp // B
    b = temp % B
    
    # Compute sum of exponentials for this row (over j dimension)
    # This requires reducing over all J values for fixed (b, h, w)
    # For simplicity, we do an atomic add approach or parallel reduction
    
    # Actually, we need to first compute the denominator for each (b, h, w)
    # Let me restructure: compute denominator per (b,h,w) position
    
    # For each (b,h,w), sum exp over j in [0, 2*C)
    # We'll compute this using a reduction
    
    # Index into exp values
    exp_base = b * out_s0 + h * out_s1 + w * out_s2
    
    # Compute sum of exponentials
    denom = 0.0
    for jj in range(C * 2):
        exp_idx = exp_base + jj * out_s3
        exp_val = tl.load(exp_idx)
        denom += exp_val
    
    # Now normalize
    exp_ptr = softmax_out_ptr + exp_base + j * out_s3
    exp_val = tl.load(exp_ptr, mask=mask)
    normalized = exp_val / denom
    
    # Store to softmax output
    tl.store(exp_ptr, normalized, mask=mask)
    
    # For sliced output (first C elements only)
    sliced_ptr = sliced_out_ptr + exp_base + j * out_s3
    sliced_val = tl.where(j < C, normalized, 0.0)
    tl.store(sliced_ptr, sliced_val, mask=mask)


@triton.jit  
def fused_einsum_cat_softmax_optimized_kernel(
    energy_ptr, query_ptr, key_ptr, softmax_out_ptr, sliced_out_ptr,
    B: tl.constexpr, H: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
    energy_s0, energy_s1, energy_s2, energy_s3,
    query_s0, query_s1, query_s2, query_s3,
    key_s0, key_s1, key_s2, key_s3,
    out_s0, out_s1, out_s2, out_s3,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized fused kernel: einsum + cat + softmax in one pass.
    Uses shared memory for intermediate values.
    """
    pid = tl.program_id(0)
    total_positions = B * H * W  # Number of (b,h,w) positions
    total_elements = total_positions * C
    block_start = pid * BLOCK_SIZE
    block_end = block_start + BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements - block_start
    base_idx = block_start + offsets
    
    # Compute indices: each thread handles one (b,h,w,j) position
    j = base_idx // total_positions
    temp = base_idx % total_positions
    w = temp // (B * H)
    temp = temp % (B * H)
    h = temp // B
    b = temp % B
    
    # Load energy[b, h, w, j]
    energy_idx = b * energy_s0 + h * energy_s1 + w * energy_s2 + j * energy_s3
    energy_val = tl.load(energy_ptr + energy_idx, mask=mask, other=0.0)
    
    # Compute einsum: sum_c query[b,c,h,w] * key[b,c,h,j]
    einsum_val = tl.cast(0.0, tl.float32)
    for c_idx in range(C):
        q_idx = b * query_s0 + c_idx * query_s1 + h * query_s2 + w * query_s3
        k_idx = b * key_s0 + c_idx * key_s1 + h * key_s2 + j * key_s3
        q_val = tl.load(query_ptr + q_idx, mask=mask, other=0.0)
        k_val = tl.load(key_ptr + k_idx, mask=mask, other=0.0)
        einsum_val += q_val * k_val
    
    # Concatenate: energy for j < C, einsum for j >= C
    concat_val = tl.where(j < C, energy_val, einsum_val)
    
    # Compute exp
    exp_val = tl.exp(concat_val)
    
    # Store exp values
    out_idx = b * out_s0 + h * out_s1 + w * out_s2 + j * out_s3
    tl.store(softmax_out_ptr + out_idx, exp_val, mask=mask)


@triton.jit
def softmax_denom_kernel(
    exp_ptr, denom_ptr,
    B: tl.constexpr, H: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
    exp_s0, exp_s1, exp_s2, exp_s3,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute denominator (sum of exp) for each (b,h,w) position.
    Each thread block handles one (b,h,w) position and reduces over j.
    """
    pid = tl.program_id(0)
    total_positions = B * H * W
    
    b = pid // (H * W)
    temp = pid % (H * W)
    h = temp // W
    w = temp % W
    
    # Compute base index
    base_idx = b * exp_s0 + h * exp_s1 + w * exp_s2
    
    # Reduce over j dimension
    denom = tl.cast(0.0, tl.float32)
    for j in range(C * 2):
        exp_idx = base_idx + j * exp_s3
        exp_val = tl.load(exp_ptr + exp_idx)
        denom += exp_val
    
    # Store denominator
    tl.store(denom_ptr + pid, denom)


@triton.jit
def softmax_normalize_final_kernel(
    exp_ptr, denom_ptr, softmax_out_ptr, sliced_out_ptr,
    B: tl.constexpr, H: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
    exp_s0, exp_s1, exp_s2, exp_s3,
    out_s0, out_s1, out_s2, out_s3,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Normalize exp values by denominator and store final results.
    Each thread handles one (b,h,w,j) position.
    """
    pid = tl.program_id(0)
    total_positions = B * H * W
    total_elements = total_positions * C
    block_start = pid * BLOCK_SIZE
    block_end = block_start + BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements - block_start
    base_idx = block_start + offsets
    
    # Compute indices
    j = base_idx // total_positions
    temp = base_idx % total_positions
    w = temp // (B * H)
    temp = temp % (B * H)
    h = temp // B
    b = temp % B
    
    # Get denominator for this (b,h,w) position
    pos_id = b * H * W + h * W + w
    denom = tl.load(denom_ptr + pos_id)
    
    # Normalize
    out_idx = b * out_s0 + h * out_s1 + w * out_s2 + j * out_s3
    exp_val = tl.load(exp_ptr + out_idx, mask=mask)
    normalized = exp_val / denom
    tl.store(softmax_out_ptr + out_idx, normalized, mask=mask)
    
    # For sliced output, store only first C elements
    sliced_val = tl.where(j < C, normalized, 0.0)
    tl.store(sliced_out_ptr + out_idx, sliced_val, mask=mask)


@torch.fx.wrap
def fused_einsum_cat_softmax_kernel_wrapper(energy, query, key):
    """
    Wrapper for the fused einsum + cat + softmax kernel.
    
    Args:
        energy: [B, H, W, C] - in_0 (energy tensor)
        query: [B, C, H, W] - in_2 (query tensor)
        key: [B, C, H, C] - in_1 (key tensor)
    
    Returns:
        softmax_out: [B, H, W, 2*C] - full softmax output
        sliced_out: [B, H, W, C] - first C elements
    """
    B, H, W, C = energy.shape
    J = C  # key's last dimension
    
    # Allocate output tensors
    softmax_out = torch.empty((B, H, W, 2 * C), dtype=torch.float32, device=energy.device)
    sliced_out = torch.empty((B, H, W, C), dtype=torch.float32, device=energy.device)
    
    # Get strides
    energy_s0, energy_s1, energy_s2, energy_s3 = energy.stride()
    query_s0, query_s1, query_s2, query_s3 = query.stride()
    key_s0, key_s1, key_s2, key_s3 = key.stride()
    out_s0, out_s1, out_s2, out_s3 = softmax_out.stride()
    
    # Convert to float32 for computation
    energy_f32 = energy.to(torch.float32)
    query_f32 = query.to(torch.float32)
    key_f32 = key.to(torch.float32)
    
    # Re-compute strides after conversion
    energy_s0, energy_s1, energy_s2, energy_s3 = energy_f32.stride()
    query_s0, query_s1, query_s2, query_s3 = query_f32.stride()
    key_s0, key_s1, key_s2, key_s3 = key_f32.stride()
    
    BLOCK_SIZE = 64
    grid_size = (B * H * W + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # First kernel: compute exp values
    fused_einsum_cat_softmax_optimized_kernel[(grid_size,)](
        energy_f32, query_f32, key_f32, softmax_out, sliced_out,
        B, H, W, C,
        energy_s0, energy_s1, energy_s2, energy_s3,
        query_s0, query_s1, query_s2, query_s3,
        key_s0, key_s1, key_s2, key_s3,
        out_s0, out_s1, out_s2, out_s3,
        BLOCK_SIZE,
    )
    
    # Allocate denominator tensor
    denom = torch.empty((B * H * W,), dtype=torch.float32, device=energy.device)
    
    # Second kernel: compute denominators
    denom_grid = (B * H * W,)
    softmax_denom_kernel[(denom_grid,)](
        softmax_out, denom,
        B, H, W, C,
        out_s0, out_s1, out_s2, out_s3,
        BLOCK_SIZE,
    )
    
    # Third kernel: normalize and store final results
    softmax_normalize_final_kernel[(grid_size,)](
        softmax_out, denom, softmax_out, sliced_out,
        B, H, W, C,
        out_s0, out_s1, out_s2, out_s3,
        out_s0, out_s1, out_s2, out_s3,
        BLOCK_SIZE,
    )
    
    # Convert back to original dtype
    softmax_out = softmax_out.to(energy.dtype)
    sliced_out = sliced_out.to(energy.dtype)
    
    return softmax_out, sliced_out


def pattern(in_0, in_1, in_2):
    """
    Match the pattern: einsum -> cat -> softmax -> slice
    """
    einsum = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
    tmp_2 = torch.cat([in_0, einsum], dim=-1)
    tmp_3 = torch.nn.functional.softmax(tmp_2, dim=-1)
    tmp_4 = tmp_3[(Ellipsis, slice(None, 64, None))]
    return tmp_3, tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_einsum_cat_softmax_kernel_wrapper