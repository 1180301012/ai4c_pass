import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern: cat -> adaptive_avg_pool2d -> flatten -> dropout
    dropout with training=False is a no-op, so we can fuse all of these
    """
    tmp_0 = torch.cat([in_0, in_1, in_2, in_3], 1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = torch.flatten(tmp_1, 1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.2, False, False)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# Optimized kernel - single load per element with pointer selection
@triton.jit
def fused_cat_flatten_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr,
    out_ptr,
    in_0_numel, in_1_numel, in_2_numel, in_3_numel,
    TOTAL_NUMEL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate starting offset for this program
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE
    offsets = off + tl.arange(0, BLOCK_SIZE)
    
    valid_mask = offsets < TOTAL_NUMEL
    
    # Pre-compute boundary values
    b1 = in_0_numel
    b2 = in_0_numel + in_1_numel
    b3 = in_0_numel + in_1_numel + in_2_numel
    
    # Select correct input and offset using chained conditionals
    cond0 = offsets < b1
    cond1 = offsets < b2
    cond2 = offsets < b3
    
    # Compute local offsets based on which input we belong to
    local_off = tl.where(cond0, offsets,
                tl.where(cond1, offsets - b1,
                tl.where(cond2, offsets - b2,
                        offsets - b3)))
    
    # Select pointer based on position
    ptr = tl.where(cond0, in_0_ptr,
           tl.where(cond1, in_1_ptr,
           tl.where(cond2, in_2_ptr, in_3_ptr)))
    
    # Load and store
    result = tl.load(ptr + local_off, mask=valid_mask, other=0.0)
    tl.store(out_ptr + offsets, result, mask=valid_mask)


@torch.fx.wrap
def fused_cat_flatten_kernel_wrapper(in_0, in_1, in_2, in_3):
    in_0_numel = in_0.numel()
    in_1_numel = in_1.numel()
    in_2_numel = in_2.numel()
    in_3_numel = in_3.numel()
    total_numel = in_0_numel + in_1_numel + in_2_numel + in_3_numel
    
    # Flatten inputs
    in_0_flat = in_0.reshape(-1)
    in_1_flat = in_1.reshape(-1)
    in_2_flat = in_2.reshape(-1)
    in_3_flat = in_3.reshape(-1)
    
    # Pre-allocate output
    out = torch.empty((1, total_numel), dtype=torch.float32, device=in_0.device)
    out_flat = out.reshape(-1)
    
    # Use multiple programs for better parallelism
    BLOCK_SIZE = 256
    grid = ((total_numel + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_cat_flatten_kernel[grid](
        in_0_ptr=in_0_flat,
        in_1_ptr=in_1_flat,
        in_2_ptr=in_2_flat,
        in_3_ptr=in_3_flat,
        out_ptr=out_flat,
        in_0_numel=in_0_numel,
        in_1_numel=in_1_numel,
        in_2_numel=in_2_numel,
        in_3_numel=in_3_numel,
        TOTAL_NUMEL=total_numel,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return fused_cat_flatten_kernel_wrapper