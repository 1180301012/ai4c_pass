import torch
import triton
import triton.language as tl


def pattern(tmp_2):
    """
    Match permute + contiguous + view for graph 0: view(1, 64, 32)
    """
    tmp_3 = tmp_2.permute(0, 2, 1, 3)
    tmp_4 = tmp_3.contiguous()
    tmp_5 = tmp_4.view(1, 64, 32)
    return tmp_5


def replacement_args(x):
    """
    Extract arguments needed for replacement.
    """
    return (x,)


@triton.jit
def fused_reshape_kernel(
    input_ptr, output_ptr,
    B: tl.constexpr, H: tl.constexpr, C: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs permute + contiguous + view in a single kernel.
    Input shape: [B, C, H, W]
    Output shape: [B, H, C*W]
    
    Operation: output[b, h, cw] = input[b, cw // W, h, cw % W]
    which is equivalent to permute(0, 2, 1, 3) + view
    """
    # Total output elements
    num_elements = B * H * C * W
    
    # Get program and compute offset
    pid = tl.program_id(0)
    elements_per_program = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    start = pid * elements_per_program
    
    # Create offset range for this program
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Convert flat output index to 3D [B, H, C*W]
    b_out = offsets // (H * C * W)
    remainder = offsets % (H * C * W)
    h_out = remainder // (C * W)
    cw_out = remainder % (C * W)
    
    # Map to input indices [B, C, H, W]
    # After permute(0, 2, 1, 3), output[b, h, c, w] = input[b, c, h, w]
    c_in = cw_out // W
    w_in = cw_out % W
    
    # Compute input flat index: [B, C, H, W] with strides [C*H*W, H*W, W, 1]
    input_offset = b_out * (C * H * W) + c_in * (H * W) + h_out * W + w_in
    
    # Load and store
    x = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def fused_reshape_wrapper(x):
    """
    Wrapper function that performs the optimized operation.
    Replaces: permute + contiguous + view with a fused Triton kernel.
    """
    # Get input shape
    B, C, H, W = x.shape
    out_shape = (B, H, C * W)
    
    # Allocate output tensor
    output = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Calculate total elements
    num_elements = B * H * C * W
    BLOCK_SIZE = 1024
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_reshape_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=output,
        B=B, H=H, C=C, W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    """
    Returns the replacement function.
    """
    return fused_reshape_wrapper