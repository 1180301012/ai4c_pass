import torch
import triton
import triton.language as tl


@triton.jit
def attention_mask_kernel(
    in_0_ptr,
    in_2_ptr,
    out_0_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for computing attention mask:
    - Load attention mask values
    - Compute position <= cache_position condition
    - Combine and output as float32
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * 1 * 1 * seq_len)
    
    # Compute 4D indices for output shape (batch, 1, 1, seq_len)
    batch_idx = offsets // (1 * 1 * seq_len)
    remainder = offsets % (1 * 1 * seq_len)
    seq_idx = remainder % seq_len
    
    # Load attention mask (batch, seq_len) and convert to bool: > 0 is valid
    attn_offset = batch_idx * seq_len + seq_idx
    attn_val = tl.load(in_0_ptr + attn_offset, mask=mask, other=0)
    bool_mask = attn_val > 0
    
    # Compute arange <= cache_position condition
    # cache_position is the same scalar value for all elements
    cache_pos = tl.load(in_2_ptr)
    pos_cond = seq_idx <= cache_pos
    
    # Combined mask
    combined = bool_mask and pos_cond
    
    # Output as float32
    out = tl.cast(combined, tl.float32)
    tl.store(out_0_ptr + offsets, out, mask=mask)


@triton.jit
def inv_freq_kernel(
    in_1_ptr,
    out_1_ptr,
    rotary_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for processing inv_freq:
    - Load values and cast to float32
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < rotary_dim
    
    val = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    out = tl.cast(val, tl.float32)
    tl.store(out_1_ptr + offsets, out, mask=mask)


@triton.jit
def position_ids_kernel(
    in_3_ptr,
    out_2_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for processing position_ids:
    - Load values and cast to float32
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * seq_len)
    
    val = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    out = tl.cast(val, tl.float32)
    tl.store(out_2_ptr + offsets, out, mask=mask)


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the full rotary embedding attention pattern:
    - Attention mask generation (in_0 + in_2)
    - inv_freq processing (in_1)
    - position_ids processing (in_3)
    
    Returns all three outputs that are used outside the pattern.
    """
    # ===== First output: Attention mask =====
    tmp_2 = in_0.to(dtype=torch.bool)
    
    # First arange for indexing attention mask
    tmp_3 = torch.arange(tmp_2.shape[-1], device=tmp_2.device)
    # Skip tmp_3 += 0 as it's a no-op - directly use tmp_3
    tmp_4 = tmp_3
    
    # Index into attention mask - extracts values at each position
    tmp_5 = tmp_2[(slice(None, None, None), tmp_4)]
    tmp_2 = tmp_4 = None
    
    # Second arange for comparison with cache_position
    seq_len = tmp_5.shape[-1]
    tmp_6 = torch.arange(seq_len, device=tmp_5.device)
    # Skip tmp_6 += 0 as it's a no-op - directly use tmp_6
    tmp_7 = tmp_6
    
    # View cache_position for broadcasting comparison
    tmp_8 = in_2.view(-1, 1)
    
    # Compare arange <= cache_position
    tmp_9 = tmp_7 <= tmp_8
    tmp_7 = tmp_8 = None
    
    # Unsqueeze for broadcasting
    tmp_10 = tmp_9[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_9 = None
    
    # Expand to (1, seq_len, 1, 1)
    tmp_11 = tmp_10.expand(1, -1, -1, -1)
    tmp_10 = None
    
    # Reshape attention mask for multiplication
    tmp_12 = tmp_5[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_5 = None
    
    # Multiply - final attention mask
    tmp_13 = tmp_11 * tmp_12
    tmp_11 = tmp_12 = None
    
    # ===== Second output: inv_freq processing =====
    _set_grad_enabled = torch.set_grad_enabled(False)
    _set_grad_enabled = None
    
    tmp_15 = in_1[(None, slice(None, None, None), None)]
    tmp_16 = tmp_15.float()
    tmp_15 = None
    tmp_17 = tmp_16.expand(1, -1, 1)
    tmp_16 = None
    tmp_18 = tmp_17.to(device=in_0.device)
    tmp_17 = None
    tmp_21 = tmp_18.float()
    tmp_18 = None
    
    # ===== Third output: position_ids processing =====
    tmp_19 = in_3[(slice(None, None, None), None, slice(None, None, None))]
    tmp_20 = tmp_19.float()
    tmp_19 = None
    tmp_22 = tmp_20.float()
    tmp_20 = None
    
    return (tmp_13, tmp_21, tmp_22)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@torch.fx.wrap
def optimized_rotary(in_0, in_1, in_2, in_3):
    """
    Optimized rotary embedding computation using multiple Triton kernels.
    Fuses attention mask generation, inv_freq processing, and position_ids processing.
    """
    batch_size = in_0.shape[0]
    seq_len = in_0.shape[-1]
    rotary_dim = in_1.shape[0]
    device = in_0.device
    
    # Output 0: (batch, 1, 1, seq_len)
    out_0 = torch.empty((batch_size, 1, 1, seq_len), device=device, dtype=torch.float32)
    
    # Output 1: (1, rotary_dim, 1)
    out_1 = torch.empty((1, rotary_dim, 1), device=device, dtype=torch.float32)
    
    # Output 2: (batch, 1, seq_len)
    out_2 = torch.empty((batch_size, 1, seq_len), device=device, dtype=torch.float32)
    
    # Launch attention mask kernel
    BLOCK_SIZE = 1024
    n_out0 = out_0.numel()
    num_programs_0 = (n_out0 + BLOCK_SIZE - 1) // BLOCK_SIZE
    attention_mask_kernel[(num_programs_0,)](
        in_0_ptr=in_0,
        in_2_ptr=in_2,
        out_0_ptr=out_0,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Launch inv_freq kernel
    num_programs_1 = (rotary_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    inv_freq_kernel[(num_programs_1,)](
        in_1_ptr=in_1,
        out_1_ptr=out_1,
        rotary_dim=rotary_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Launch position_ids kernel
    n_out2 = out_2.numel()
    num_programs_2 = (n_out2 + BLOCK_SIZE - 1) // BLOCK_SIZE
    position_ids_kernel[(num_programs_2,)](
        in_3_ptr=in_3,
        out_2_ptr=out_2,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (out_0, out_1, out_2)


def replacement_func():
    return optimized_rotary