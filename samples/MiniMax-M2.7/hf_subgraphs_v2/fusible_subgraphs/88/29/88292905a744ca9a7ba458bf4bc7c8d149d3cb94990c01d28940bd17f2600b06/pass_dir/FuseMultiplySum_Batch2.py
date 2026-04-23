import torch
import triton
import triton.language as tl


@triton.jit
def fused_multiply_sum_kernel(
    weight_ptr,
    in_0_ptr,
    output_ptr,
    batch_size,
    num_channels,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes:
    output[b, s] = sum over c of (weight[b, c] * in_0[b, c, s])
    
    This fuses:
    - element-wise multiplication: tmp_4 = tmp_3 * in_0
    - sum reduction: tmp_5 = sum(tmp_4, dim=1)
    
    The key optimization is avoiding materializing tmp_4 entirely.
    """
    # Grid: one program per (batch, spatial_position)
    pid = tl.program_id(0)
    
    # Compute batch and spatial indices
    num_spatial = batch_size * spatial_size
    b = pid // spatial_size
    s = pid % spatial_size
    
    # Accumulator for sum
    acc = 0.0
    
    # Loop over channels
    for c in range(num_channels):
        # Compute weight offset: weight[b, c] with shape [B, C]
        weight_offset = b * num_channels + c
        w = tl.load(weight_ptr + weight_offset)
        
        # Compute in_0 offset: in_0[b, c, s] with shape [B, C, S]
        in_0_offset = (b * num_channels * spatial_size + 
                       c * spatial_size + 
                       s)
        x = tl.load(in_0_ptr + in_0_offset)
        
        # Accumulate: w * x
        acc += w * x
    
    # Store result
    out_offset = b * spatial_size + s
    tl.store(output_ptr + out_offset, acc)


@torch.fx.wrap
def fused_multiply_sum(weight, in_0):
    """
    Fused multiply and sum operation.
    
    Args:
        weight: [B, C] - attention weights after softmax
        in_0: [B, C, S1, S2, ...] - input tensor with C channels
    
    Returns:
        output: [B, S1, S2, ...] - weighted sum across channels
    """
    B = weight.shape[0]
    C = weight.shape[1]
    
    # Flatten spatial dimensions
    orig_shape = in_0.shape
    in_0_flat = in_0.view(B, C, -1)
    spatial_size = in_0_flat.shape[2]
    
    # Allocate output
    output = torch.empty((B, spatial_size), dtype=in_0.dtype, device=in_0.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = B * spatial_size
    
    fused_multiply_sum_kernel[(num_programs,)](
        weight_ptr=weight,
        in_0_ptr=in_0_flat,
        output_ptr=output,
        batch_size=B,
        num_channels=C,
        spatial_size=spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to original spatial shape
    return output.view(orig_shape[0], *orig_shape[2:])


def pattern(in_0, in_1):
    """
    Match the exact pattern from model.py with batch size 2:
    softmax(in_1, dim=1) -> reshape(2, -1) -> view(2, -1, 1, 1) -> view(2, 2, -1, 1, 1)
    -> multiply with in_0 -> sum(dim=1) -> contiguous
    """
    tmp_0 = torch.nn.functional.softmax(in_1, dim=1)
    tmp_1 = tmp_0.reshape(2, -1)
    tmp_2 = tmp_1.view(2, -1, 1, 1)
    tmp_3 = tmp_2.view(2, 2, -1, 1, 1)
    tmp_4 = tmp_3 * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_multiply_sum