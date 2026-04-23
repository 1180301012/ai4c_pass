import torch
import triton
import triton.language as tl

@triton.jit
def bilinear_interpolate_kernel(
    input_ptr,
    output_ptr,
    batch_size, channels, in_height, in_width,
    out_height, out_width,
    scale_h, scale_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_elements = batch_size * channels * out_height * out_width
    output_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = output_idx < n_elements
    
    tmp = output_idx
    out_w = tmp % out_width
    tmp = tmp // out_width
    out_h = tmp % out_height
    tmp = tmp // out_height
    ch = tmp % channels
    batch = tmp // channels
    
    in_h_float = out_h.to(tl.float32) / scale_h.to(tl.float32)
    in_w_float = out_w.to(tl.float32) / scale_w.to(tl.float32)
    
    in_h0 = tl.floor(in_h_float).to(tl.int32)
    in_w0 = tl.floor(in_w_float).to(tl.int32)
    in_h1 = in_h0 + 1
    in_w1 = in_w0 + 1
    
    in_h0_clamped = tl.maximum(tl.minimum(in_h0, in_height - 1), 0)
    in_w0_clamped = tl.maximum(tl.minimum(in_w0, in_width - 1), 0)
    in_h1_clamped = tl.maximum(tl.minimum(in_h1, in_height - 1), 0)
    in_w1_clamped = tl.maximum(tl.minimum(in_w1, in_width - 1), 0)
    
    h1_weight = in_h_float - in_h0.to(tl.float32)
    w1_weight = in_w_float - in_w0.to(tl.float32)
    h0_weight = 1.0 - h1_weight
    w0_weight = 1.0 - w1_weight
    
    base_idx = ((batch * channels + ch) * in_height)
    
    idx_00 = base_idx + in_h0_clamped * in_width + in_w0_clamped
    inp_00 = tl.load(input_ptr + idx_00, mask=mask, other=0.0)
    
    idx_01 = base_idx + in_h0_clamped * in_width + in_w1_clamped
    inp_01 = tl.load(input_ptr + idx_01, mask=mask, other=0.0)
    
    idx_10 = base_idx + in_h1_clamped * in_width + in_w0_clamped
    inp_10 = tl.load(input_ptr + idx_10, mask=mask, other=0.0)
    
    idx_11 = base_idx + in_h1_clamped * in_width + in_w1_clamped
    inp_11 = tl.load(input_ptr + idx_11, mask=mask, other=0.0)
    
    out = (inp_00 * h0_weight * w0_weight + 
           inp_01 * h0_weight * w1_weight +
           inp_10 * h1_weight * w0_weight +
           inp_11 * h1_weight * w1_weight)
    
    tl.store(output_ptr + output_idx, out, mask=mask)


@torch.fx.wrap
def bilinear_interpolate_wrapper(input_tensor):
    B, C, in_H, in_W = input_tensor.shape
    # Hardcoded parameters from the pattern
    out_H, out_W = 512, 512
    
    output = torch.empty((B, C, out_H, out_W), dtype=input_tensor.dtype, device=input_tensor.device)
    
    scale_h = out_H / in_H
    scale_w = out_W / in_W
    
    n_elements = B * C * out_H * out_W
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    bilinear_interpolate_kernel[(num_programs,)](
        input_tensor,
        output,
        B, C, in_H, in_W,
        out_H, out_W,
        scale_h, scale_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(conv_output):
    return torch.nn.functional.interpolate(conv_output, size = (512, 512), mode = 'bilinear', align_corners = False)


def replacement_args(conv_output):
    return (conv_output,)


def replacement_func():
    return bilinear_interpolate_wrapper