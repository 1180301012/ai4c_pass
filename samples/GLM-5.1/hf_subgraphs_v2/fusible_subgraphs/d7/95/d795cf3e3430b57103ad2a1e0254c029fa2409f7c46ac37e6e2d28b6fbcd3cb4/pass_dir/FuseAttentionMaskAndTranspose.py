import torch
import triton
import triton.language as tl

# Pattern: match the unsqueeze-subtract-comparison-masked_fill chain
# that takes tmp_9 (shape 1,361,49) and produces the final attention mask tmp_16
def pattern(tmp_9):
    tmp_10 = tmp_9.unsqueeze(2)
    tmp_11 = tmp_9.unsqueeze(3)
    tmp_12 = tmp_10 - tmp_11
    tmp_13 = tmp_12 != 0
    tmp_14 = tmp_12.masked_fill(tmp_13, -1000.0)
    tmp_15 = tmp_12 == 0
    tmp_16 = tmp_14.masked_fill(tmp_15, 0.0)
    return tmp_16

def replacement_args(tmp_9):
    return (tmp_9,)


@triton.jit
def attention_mask_kernel(
    mask_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel that directly computes the attention mask from the 1,361,49 mask values.
    
    The mask values represent which positions in the 133x133 grid are boundary (1) or normal (0).
    The output is: 0.0 where both positions have the same mask value, -1000.0 where different.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    valid = offsets < N
    
    # Shape is (1, 361, 49, 49), total elements = 361 * 49 * 49 = 864721
    # Decompose: ac = offsets // (49*49), rs1 = (offsets % (49*49)) // 49, rs2 = offsets % 49
    RS_AREA = 49 * 49  # = 2401
    ac = offsets // RS_AREA
    rs_rem = offsets % RS_AREA
    rs1 = rs_rem // 49
    rs2 = rs_rem % 49
    
    # Load the two mask values
    # mask_ptr has shape (1, 361, 49), so idx = ac * 49 + rs
    val1_idx = ac * 49 + rs1
    val2_idx = ac * 49 + rs2
    
    val1 = tl.load(mask_ptr + val1_idx, mask=valid, other=0.0)
    val2 = tl.load(mask_ptr + val2_idx, mask=valid, other=0.0)
    
    # If val1 != val2, output -1000.0; else 0.0
    # This is equivalent to: masked_fill(where != 0, -1000.0) then masked_fill(where == 0, 0.0)
    # But since val1-val2 != 0 ↔ val1 != val2, and val1-val2 == 0 ↔ val1 == val2
    different = val1 != val2
    result = tl.where(different, -1000.0, 0.0)
    
    tl.store(out_ptr + offsets, result, mask=valid)


@torch.fx.wrap
def compute_attention_mask(tmp_9):
    """Compute the attention mask from the window mask tensor."""
    dtype = tmp_9.dtype
    device = tmp_9.device
    
    N = 361 * 49 * 49  # Total elements in output
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Output tensor - use float32 for the mask values (as in original model)
    # The original uses -1000.0 and 0.0 which are float values
    out = torch.empty((1, 361, 49, 49), dtype=torch.float32, device=device)
    
    attention_mask_kernel[(num_programs,)](
        mask_ptr=tmp_9,
        out_ptr=out,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return compute_attention_mask