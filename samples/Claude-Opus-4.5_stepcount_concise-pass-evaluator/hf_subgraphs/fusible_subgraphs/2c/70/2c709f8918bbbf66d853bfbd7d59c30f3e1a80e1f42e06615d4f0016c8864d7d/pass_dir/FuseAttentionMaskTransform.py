import torch
import triton
import triton.language as tl

# Pattern to match: in_1.to(float32) -> (1.0 - x) -> x * large_negative_constant
def pattern(in_1):
    tmp_1 = in_1.to(dtype=torch.float32)
    tmp_2 = 1.0 - tmp_1
    tmp_3 = tmp_2 * -3.4028234663852886e+38
    return tmp_3

# Extract arguments from matched pattern
def replacement_args(in_1):
    return (in_1,)

# Triton kernel for fused attention mask transformation
@triton.jit
def fused_attention_mask_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load as int64 and convert to float32
    x = tl.load(in_ptr + offsets, mask=mask).to(tl.float32)
    
    # Fused computation: (1.0 - x) * -3.4028234663852886e+38
    result = (1.0 - x) * -3.4028234663852886e+38
    
    # Store
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_attention_mask(in_1):
    n_elements = in_1.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty(in_1.shape, dtype=torch.float32, device=in_1.device)
    
    fused_attention_mask_kernel[(num_programs,)](
        in_ptr=in_1,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_attention_mask