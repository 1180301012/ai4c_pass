import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Match: linear -> sigmoid -> view(128, 64, 1, 1) -> mul pattern
    Returns view output and final multiplication result
    """
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.sigmoid(linear)
    tmp_4 = tmp_3.view(128, 64, 1, 1)
    tmp_5 = in_3 * tmp_4
    return tmp_4, tmp_5

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_sigmoid_mul_kernel(
    linear_ptr,
    in_3_ptr,
    output_ptr,
    n_elements,
    linear_numel,
    ch_per_batch,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute linear index
    # For in_3[b, ch, h, w], flat_idx = b*64*spatial_size + ch*spatial_size + local
    # linear[b, ch] = linear[b*64 + ch]
    batch_idx = offsets // ch_per_batch
    ch_idx = (offsets % ch_per_batch) // spatial_size
    linear_idx = batch_idx * 64 + ch_idx
    linear_mask = linear_idx < linear_numel
    
    # Load linear output
    x = tl.load(linear_ptr + linear_idx, mask=linear_mask, other=0.0)
    
    # Compute sigmoid
    exp_neg_x = tl.exp(-x)
    sig = 1.0 / (1.0 + exp_neg_x)
    
    # Load in_3 and multiply
    y = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    out = sig * y
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_sigmoid_mul(linear_out, in_3):
    B, C, H, W = in_3.shape
    N = in_3.numel()
    
    BLOCK_SIZE = 512
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_3)
    ch_per_batch = C * H * W
    spatial_size = H * W
    
    fused_sigmoid_mul_kernel[(num_programs,)](
        linear_ptr=linear_out,
        in_3_ptr=in_3,
        output_ptr=out,
        n_elements=N,
        linear_numel=linear_out.numel(),
        ch_per_batch=ch_per_batch,
        spatial_size=spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_sigmoid_mul