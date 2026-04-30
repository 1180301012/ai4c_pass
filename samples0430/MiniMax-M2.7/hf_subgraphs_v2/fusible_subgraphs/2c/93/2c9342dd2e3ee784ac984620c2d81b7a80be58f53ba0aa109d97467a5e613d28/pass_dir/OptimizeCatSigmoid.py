import torch
import triton
import triton.language as tl

@triton.jit
def cat_sigmoid_scale_kernel(
    in3_ptr,
    in4_ptr,
    conv_ptr,
    output_ptr,
    n_in3,
    n_in4,
    n_conv,
    total_elements,
    PI: tl.constexpr,
    PI_OVER_4: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused cat + sigmoid + scale kernel.
    Computes: concat(in3, in4, conv_view) -> sigmoid -> scale
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Determine which input this element comes from
    # Layout: [in3 (n_in3), in4 (n_in4), conv (n_conv)]
    result = tl.zeros((BLOCK_SIZE,), tl.float32)
    
    # Process in3 elements
    in3_mask = offsets < n_in3
    in3_offsets = offsets
    x = tl.load(in3_ptr + in3_offsets, mask=in3_mask & mask, other=0.0).to(tl.float32)
    
    # Process in4 elements
    in4_start = n_in3
    in4_end = n_in3 + n_in4
    in4_mask = (offsets >= in4_start) & (offsets < in4_end)
    in4_offsets = offsets - in4_start
    x = tl.where(in4_mask, tl.load(in4_ptr + in4_offsets, mask=in4_mask & mask, other=0.0).to(tl.float32), x)
    
    # Process conv elements
    conv_start = n_in3 + n_in4
    conv_mask = offsets >= conv_start
    conv_offsets = offsets - conv_start
    x = tl.where(conv_mask, tl.load(conv_ptr + conv_offsets, mask=conv_mask & mask, other=0.0).to(tl.float32), x)
    
    # Apply sigmoid and scale
    x_sigmoid = 1.0 / (1.0 + tl.exp(-x))
    out = x_sigmoid * PI - PI_OVER_4
    
    tl.store(output_ptr + offsets, out.to(tl.bfloat16), mask=mask)


@torch.fx.wrap
def cat_sigmoid_scale_wrapper(in3, in4, conv_view):
    """
    Fused cat + sigmoid + scale operation.
    """
    n_in3 = in3.numel()
    n_in4 = in4.numel()
    n_conv = conv_view.numel()
    total_elements = n_in3 + n_in4 + n_conv
    
    BLOCK_SIZE = 256
    num_programs = max(1, (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    output = torch.empty((total_elements,), dtype=in3.dtype, device=in3.device)
    
    cat_sigmoid_scale_kernel[(num_programs,)](
        in3_ptr=in3,
        in4_ptr=in4,
        conv_ptr=conv_view,
        output_ptr=output,
        n_in3=n_in3,
        n_in4=n_in4,
        n_conv=n_conv,
        total_elements=total_elements,
        PI=3.141592653589793,
        PI_OVER_4=0.785398163397448,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to match expected output shape
    output = output.reshape(in3.shape[0], 1, -1)
    return output


def pattern(in_3, in_4, tmp_3):
    """
    Match the pattern: view -> cat -> sigmoid -> sub -> mul
    Returns: tmp_7 (the final result)
    """
    tmp_4 = torch.cat([in_3, in_4, tmp_3], 2)
    tmp_5 = tmp_4.sigmoid()
    tmp_6 = tmp_5 - 0.25
    tmp_7 = tmp_6 * 3.141592653589793
    return tmp_7


def replacement_args(in_3, in_4, tmp_3):
    return (in_3, in_4, tmp_3)


def replacement_func():
    return cat_sigmoid_scale_wrapper