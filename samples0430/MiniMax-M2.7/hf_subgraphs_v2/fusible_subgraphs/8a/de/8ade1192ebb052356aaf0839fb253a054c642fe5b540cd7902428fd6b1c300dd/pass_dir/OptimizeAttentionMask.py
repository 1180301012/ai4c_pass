import torch
import triton
import triton.language as tl

def pattern(in_5):
    """
    Pattern to match: attention mask generation
    Converts mask to float, computes 1 - mask, converts to bool for masked_fill
    """
    tmp_4 = in_5.to(torch.float32)
    tmp_5 = torch.tensor(1.0, dtype=torch.float32)
    tmp_6 = tmp_5 - tmp_4
    tmp_7 = tmp_6.to(torch.bool)
    tmp_8 = tmp_6.masked_fill(tmp_7, -3.4028234663852886e+38)
    return tmp_8

def replacement_args(in_5):
    return (in_5,)

@triton.jit
def attention_mask_kernel(
    mask_ptr, output_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load mask values (int64 -> float32)
    mask_val = tl.load(mask_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute 1 - mask
    neg_mask = 1.0 - mask_val
    
    # Create boolean mask: True where neg_mask should be filled
    # masked_fill with -inf happens where the bool is True
    # So we compute: where(neg_mask > 0.5, -inf, neg_mask)
    neg_mask_filled = tl.where(neg_mask > 0.5, float('-inf'), neg_mask)
    
    tl.store(output_ptr + offsets, neg_mask_filled, mask=mask)

@torch.fx.wrap
def fused_attention_mask(in_5):
    """
    Fused attention mask generation kernel.
    Combines: to(float32) -> 1 - mask -> to(bool) -> masked_fill(-inf)
    """
    n_elements = in_5.numel()
    output = torch.empty([n_elements], dtype=torch.float32, device=in_5.device)
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    attention_mask_kernel[(num_programs,)](
        mask_ptr=in_5,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_attention_mask