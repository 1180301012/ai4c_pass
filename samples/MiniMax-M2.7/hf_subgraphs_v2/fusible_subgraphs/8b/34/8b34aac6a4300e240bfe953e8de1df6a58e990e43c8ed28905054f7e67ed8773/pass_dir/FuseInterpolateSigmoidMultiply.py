import torch
import triton
import triton.language as tl


@triton.jit
def interpolate_sigmoid_multiply_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    batch_size,
    num_channels,
    in_height,
    in_width,
    out_height,
    out_height_idx,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: interpolate (bilinear) + sigmoid + multiply
    
    input: [B, C, H, W] - the tensor to interpolate and apply sigmoid
    other: [B, C, out_H, out_W] - the tensor to multiply with sigmoid(input)
    output: [B, C, out_H, out_W]
    """
    # Calculate total number of elements
    total_elements = batch_size * num_channels * out_height * out_height_idx
    
    # Get program id
    pid = tl.program_id(0)
    
    # Calculate block start and offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Calculate indices
    # Flatten: (b * c * oh * ow)
    b = offsets // (num_channels * out_height * out_height_idx)
    remaining = offsets % (num_channels * out_height * out_height_idx)
    c = remaining // (out_height * out_height_idx)
    remaining = remaining % (out_height * out_height_idx)
    oh = remaining // out_height_idx
    ow = remaining % out_height_idx
    
    # Calculate source coordinates for bilinear interpolation
    # Scale factors
    h_scale = (in_height - 1.0) / (out_height - 1.0) if out_height > 1 else 0.0
    w_scale = (in_width - 1.0) / (out_height_idx - 1.0) if out_height_idx > 1 else 0.0
    
    h = oh * h_scale
    w = ow * w_scale
    
    h0 = tl.floor(h)
    w0 = tl.floor(w)
    h1 = h0 + 1
    w1 = w0 + 1
    
    # Clamp to valid range
    h0 = tl.clamp(h0, 0, in_height - 1)
    w0 = tl.clamp(w0, 0, in_width - 1)
    h1 = tl.clamp(h1, 0, in_height - 1)
    w1 = tl.clamp(w1, 0, in_width - 1)
    
    # Calculate interpolation weights
    h_l = h - h0
    w_l = w - w0
    h_h = 1.0 - h_l
    w_h = 1.0 - w_l
    
    # Load values from input for bilinear interpolation
    # h0, w0
    idx_h0_w0 = b * num_channels * in_height * in_width + c * in_height * in_width + h0 * in_width + w0
    v_h0_w0 = tl.load(input_ptr + idx_h0_w0, mask=mask, other=0.0)
    
    # h0, w1
    idx_h0_w1 = b * num_channels * in_height * in_width + c * in_height * in_width + h0 * in_width + w1
    v_h0_w1 = tl.load(input_ptr + idx_h0_w1, mask=mask, other=0.0)
    
    # h1, w0
    idx_h1_w0 = b * num_channels * in_height * in_width + c * in_height * in_width + h1 * in_width + w0
    v_h1_w0 = tl.load(input_ptr + idx_h1_w0, mask=mask, other=0.0)
    
    # h1, w1
    idx_h1_w1 = b * num_channels * in_height * in_width + c * in_height * in_width + h1 * in_width + w1
    v_h1_w1 = tl.load(input_ptr + idx_h1_w1, mask=mask, other=0.0)
    
    # Bilinear interpolation
    v0 = h_h * v_h0_w0 + h_l * v_h1_w0
    v1 = h_h * v_h0_w1 + h_l * v_h1_w1
    interpolated = w_h * v0 + w_l * v1
    
    # Apply sigmoid: sigmoid(x) = 1 / (1 + exp(-x))
    # Compute exp(-x) safely
    neg_interpolated = -interpolated
    exp_neg = tl.exp(neg_interpolated)
    sigmoid_val = 1.0 / (1.0 + exp_neg)
    
    # Load other and multiply
    other_idx = b * num_channels * out_height * out_height_idx + c * out_height * out_height_idx + oh * out_height_idx + ow
    other_val = tl.load(other_ptr + other_idx, mask=mask, other=0.0)
    
    result = sigmoid_val * other_val
    
    # Store result
    out_idx = b * num_channels * out_height * out_height_idx + c * out_height * out_height_idx + oh * out_height_idx + ow
    tl.store(output_ptr + out_idx, result, mask=mask)


@torch.fx.wrap
def interpolate_sigmoid_multiply_wrapper(input_tensor, other_tensor, output_size):
    """
    Fused interpolate + sigmoid + multiply operation.
    
    input_tensor: [B, C, H_in, W_in] - to be interpolated and sigmoid applied
    other_tensor: [B, C, H_out, W_out] - to multiply with sigmoid result
    output_size: tuple (H_out, W_out)
    """
    batch_size, num_channels, in_height, in_width = input_tensor.shape
    out_height, out_width = output_size
    
    # Allocate output
    output = torch.empty(
        (batch_size, num_channels, out_height, out_width),
        device=input_tensor.device,
        dtype=input_tensor.dtype
    )
    
    total_elements = batch_size * num_channels * out_height * out_width
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    interpolate_sigmoid_multiply_kernel[(num_programs,)](
        input_tensor,
        other_tensor,
        output,
        batch_size,
        num_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        BLOCK_SIZE
    )
    
    return output


def pattern(in_4, in_3):
    """
    Match the pattern: interpolate(in_4, (64, 64)) → sigmoid → in_3 * sigmoid
    Returns the result tensor.
    """
    tmp_3 = torch.nn.functional.interpolate(in_4, (64, 64), None, 'bilinear', False)
    tmp_4 = torch.sigmoid(tmp_3)
    tmp_5 = in_3 * tmp_4
    return tmp_5


def replacement_args(in_4, in_3):
    return (in_4, in_3)


def replacement_func():
    return interpolate_sigmoid_multiply_wrapper