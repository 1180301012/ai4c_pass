import torch
import triton
import triton.language as tl


# Pattern matching function - matches Conv2d(1x1) + AvgPool2d(2x2) pattern
def pattern(in_0, in_1):
    # 1x1 Conv2d: kernel_size=1, stride=1, padding=0, dilation=1, groups=1
    tmp_1 = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    # AvgPool2d: kernel_size=2, stride=2, padding=0, count_include_pad=True
    tmp_2 = torch.nn.functional.avg_pool2d(tmp_1, 2, 2, 0, False, True, None)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# Optimized Triton kernel for fused Conv1x1 + AvgPool2d
@triton.jit(num_warps=4, num_stages=3)
def fused_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch, in_c, in_h, in_w,
    out_c, out_h, out_w,
    in_stride_b, in_stride_c, in_stride_h, in_stride_w,
    w_stride_oc, w_stride_ic,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w,
):
    # 1D grid: each program computes one output position
    pid = tl.program_id(0)
    total_per_sample = out_c * out_h * out_w
    
    sample_idx = pid // total_per_sample
    remainder = pid % total_per_sample
    oc = remainder // (out_h * out_w)
    remainder = remainder % (out_h * out_w)
    h = remainder // out_w
    w = remainder % out_w
    
    # Accumulate over input channels
    acc = 0.0
    for ic in range(in_c):
        base = (sample_idx * in_stride_b + ic * in_stride_c + 
                (h * 2) * in_stride_h + (w * 2) * in_stride_w)
        
        # Load and sum 2x2 pool window
        s = tl.load(input_ptr + base).to(tl.float32)
        s = s + tl.load(input_ptr + base + in_stride_h).to(tl.float32)
        s = s + tl.load(input_ptr + base + in_stride_w).to(tl.float32)
        s = s + tl.load(input_ptr + base + in_stride_h + in_stride_w).to(tl.float32)
        
        # Multiply by weight
        w_val = tl.load(weight_ptr + oc * w_stride_oc + ic * w_stride_ic).to(tl.float32)
        acc += s * w_val
    
    # Average pool divide
    acc = acc / 4.0
    
    # Store result
    out_off = (sample_idx * out_stride_b + oc * out_stride_c + 
               h * out_stride_h + w * out_stride_w)
    tl.store(output_ptr + out_off, acc)


@torch.fx.wrap
def replacement_func_impl(in_0, in_1):
    batch, in_c, in_h, in_w = in_1.shape
    out_c = in_0.shape[0]
    out_h = in_h // 2
    out_w = in_w // 2
    
    output = torch.empty((batch, out_c, out_h, out_w), device=in_1.device, dtype=in_1.dtype)
    
    # Launch one program per output element
    total_outputs = batch * out_c * out_h * out_w
    grid = (total_outputs,)
    
    fused_kernel[grid](
        in_1, in_0, output,
        batch, in_c, in_h, in_w,
        out_c, out_h, out_w,
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_0.stride(0), in_0.stride(1),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
    )
    return output


def replacement_func():
    return replacement_func_impl