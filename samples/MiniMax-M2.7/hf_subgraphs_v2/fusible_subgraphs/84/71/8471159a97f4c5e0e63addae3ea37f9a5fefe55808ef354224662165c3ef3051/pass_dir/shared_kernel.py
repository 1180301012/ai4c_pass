import torch
import triton
import triton.language as tl

@triton.jit
def fused_sigmoid_mul_kernel(
    linear_ptr,
    in_3_ptr,
    output_ptr,
    n_elements,
    linear_numel,
    H_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute linear index for current position
    # For in_3 with shape [B, 64, H, W], flat_idx maps to in_3[b, c, h, w]
    # where b = flat_idx // (64 * H * W), c = (flat_idx % (64 * H * W)) // (H * W)
    # H_stride = 64 * H * W (stride of in_3 at dim 1)
    linear_idx = (offsets // H_stride) * 64 + ((offsets % H_stride) // (H_stride // 64))
    linear_mask = linear_idx < linear_numel
    
    # Load linear output and compute sigmoid
    x = tl.load(linear_ptr + linear_idx, mask=linear_mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    
    # Load in_3 and multiply
    y = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    out = sig * y
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_sigmoid_mul(linear_out, in_3):
    B, C, H, W = in_3.shape
    N = in_3.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_3)
    H_stride = 64 * H * W
    
    fused_sigmoid_mul_kernel[(num_programs,)](
        linear_ptr=linear_out,
        in_3_ptr=in_3,
        output_ptr=out,
        n_elements=N,
        linear_numel=linear_out.numel(),
        H_stride=H_stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out