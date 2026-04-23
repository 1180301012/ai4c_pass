import torch
import triton
import triton.language as tl


@triton.jit
def fuse_relu_scale_bias_pad_kernel(
    in_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused ReLU + Scale + Bias + Pad kernel.
    
    in_ptr: input tensor [B, C, H, W]
    scale_ptr: scalar scale pointer
    bias_ptr: scalar bias pointer
    out_ptr: output tensor [B, C, H+1, W+1]
    n_elements: total elements in output tensor
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input value
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU
    x_relu = tl.maximum(x, 0.0)
    
    # Apply scale and bias
    scale = tl.load(scale_ptr)
    bias = tl.load(bias_ptr)
    x_scaled = x_relu * scale + bias
    
    # Store result
    tl.store(out_ptr + offsets, x_scaled, mask=mask)


@torch.fx.wrap
def fuse_relu_scale_bias_pad(in_0, in_1, in_2):
    """
    Fused kernel for: relu(in_2) * in_1 + in_0, then pad by 1 on H and W.
    """
    B, C, H, W = in_2.shape
    output_H = H + 1
    output_W = W + 1
    n_elements = B * C * output_H * output_W
    
    output = torch.empty((B, C, output_H, output_W), dtype=in_2.dtype, device=in_2.device)
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create scalar tensor pointers
    scale_ptr = in_1.to(device=in_2.device)
    bias_ptr = in_0.to(device=in_2.device)
    
    fuse_relu_scale_bias_pad_kernel[(num_programs,)](
        in_ptr=in_2,
        scale_ptr=scale_ptr,
        bias_ptr=bias_ptr,
        out_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


class PatternModule(torch.nn.Module):
    def forward(self, in_0, in_1, in_2):
        tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
        tmp_3 = in_1 * tmp_2
        tmp_4 = tmp_3 + in_0
        tmp_5 = torch.nn.functional.pad(tmp_4, (0, 1, 0, 1), 'constant', None)
        return tmp_5


def pattern(in_0, in_1, in_2):
    """
    Match the pattern: relu(in_2) * in_1 + in_0, then pad with (0, 1, 0, 1).
    
    IMPORTANT: Use exact operations that match the model's FX graph.
    """
    mod = PatternModule()
    return mod.forward(in_0, in_1, in_2)


def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments needed for the replacement function.
    """
    return (in_0, in_1, in_2)


def replacement_func():
    """
    Return the fused kernel function.
    """
    return fuse_relu_scale_bias_pad