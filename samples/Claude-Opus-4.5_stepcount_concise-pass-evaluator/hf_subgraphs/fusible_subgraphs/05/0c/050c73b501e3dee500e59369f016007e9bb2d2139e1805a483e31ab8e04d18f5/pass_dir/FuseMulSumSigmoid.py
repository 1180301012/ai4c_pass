import torch
import triton
import triton.language as tl

# Pattern matching function - must match model.py exactly (without None assignments)
def pattern(in_0, in_1):
    tmp_0 = in_1 * in_0
    tmp_1 = torch.sum(tmp_0, dim=1)
    tmp_2 = tmp_1.unsqueeze(1)
    tmp_3 = torch.sigmoid(tmp_2)
    return (tmp_3,)

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Triton kernel for fused mul + sum(dim=1) + unsqueeze(1) + sigmoid
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_mul_sum_sigmoid_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    C,  # channel dimension (reduction dim)
    HW,  # H * W
    n_elements,  # B * H * W (total output elements)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles BLOCK_SIZE output positions
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate batch and hw position for each offset
    # offsets = b * HW + hw
    b = offsets // HW
    hw = offsets % HW
    
    # Accumulate sum over channel dimension
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Loop over all channels
    for c in range(C):
        # Index into input: b * (C * HW) + c * HW + hw
        idx = b * (C * HW) + c * HW + hw
        val_0 = tl.load(in_0_ptr + idx, mask=mask, other=0.0)
        val_1 = tl.load(in_1_ptr + idx, mask=mask, other=0.0)
        acc += val_0 * val_1
    
    # Apply sigmoid
    result = tl.sigmoid(acc)
    
    # Store result - output shape is [B, 1, H, W]
    # which is contiguous as [B, HW] effectively
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_mul_sum_sigmoid(in_0, in_1):
    B, C, H, W = in_0.shape
    HW = H * W
    n_elements = B * HW
    
    # Output shape is [B, 1, H, W]
    out = torch.empty((B, 1, H, W), dtype=in_0.dtype, device=in_0.device)
    
    # Ensure inputs are contiguous
    in_0 = in_0.contiguous()
    in_1 = in_1.contiguous()
    
    # Calculate grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_mul_sum_sigmoid_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        C=C,
        HW=HW,
        n_elements=n_elements,
    )
    
    return (out,)

def replacement_func():
    return fused_mul_sum_sigmoid