import torch
import triton
import triton.language as tl

# Pattern matching: add + dropout2d(training=False)
# Since dropout2d with training=False is identity, this fusion just does the add
def pattern(a, b):
    added = a + b
    result = torch.nn.functional.dropout2d(added, 0.1, False, False)
    return result

def replacement_args(a, b):
    return (a, b, "add_dropout2d")

# Triton add kernel
@triton.jit
def add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_add(x, y):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    add_kernel[(num_programs,)](
        x_ptr=x, y_ptr=y, out_ptr=out,
        n_elements=N, BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

# Triton 1x1 conv2d kernel
@triton.jit
def conv1x1_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, C_IN: tl.constexpr, H, W,
    C_OUT: tl.constexpr,
    stride_ib, stride_ic, stride_ih, stride_iw,
    stride_ob, stride_oc, stride_oh, stride_ow,
    BLOCK_HW: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_hw = pid * BLOCK_HW + tl.arange(0, BLOCK_HW)
    total_hw = B * H * W
    
    offs_b = offs_hw // (H * W)
    offs_hw_rem = offs_hw % (H * W)
    offs_h = offs_hw_rem // W
    offs_w = offs_hw_rem % W
    
    mask_hw = offs_hw < total_hw
    
    # Initialize accumulator for all output channels
    acc = tl.zeros((BLOCK_HW, C_OUT), dtype=tl.float32)
    
    # Iterate over input channels in blocks
    for ci_start in range(0, C_IN, BLOCK_CI):
        offs_ci = ci_start + tl.arange(0, BLOCK_CI)
        mask_ci = offs_ci < C_IN
        
        # Load input: [BLOCK_HW, BLOCK_CI]
        input_offsets = (
            offs_b[:, None] * stride_ib +
            offs_ci[None, :] * stride_ic +
            offs_h[:, None] * stride_ih +
            offs_w[:, None] * stride_iw
        )
        input_mask = mask_hw[:, None] & mask_ci[None, :]
        a = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0).to(tl.float32)
        
        # Load weight: [C_OUT, BLOCK_CI] then transpose for dot product
        offs_co = tl.arange(0, C_OUT)
        weight_offsets = offs_ci[:, None] + offs_co[None, :] * C_IN
        weight_mask = mask_ci[:, None]
        b = tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0).to(tl.float32)
        
        # b is [BLOCK_CI, C_OUT], a is [BLOCK_HW, BLOCK_CI]
        # tl.dot(a, b) gives [BLOCK_HW, C_OUT]
        acc += tl.dot(a, b)
    
    # Add bias
    offs_co = tl.arange(0, C_OUT)
    bias_vals = tl.load(bias_ptr + offs_co).to(tl.float32)
    acc += bias_vals[None, :]
    
    # Cast to output dtype and store
    output_vals = acc.to(OUTPUT_DTYPE)
    output_offsets = (
        offs_b[:, None] * stride_ob +
        offs_co[None, :] * stride_oc +
        offs_h[:, None] * stride_oh +
        offs_w[:, None] * stride_ow
    )
    output_mask = mask_hw[:, None]
    tl.store(output_ptr + output_offsets, output_vals, mask=output_mask)

@torch.fx.wrap
def triton_conv2d_1x1(input_feat, weight, bias):
    B, C_in, H, W = input_feat.shape
    C_out = weight.shape[0]
    
    # Determine output dtype constant
    if input_feat.dtype == torch.float32:
        OUTPUT_DTYPE = tl.float32
    elif input_feat.dtype == torch.float16:
        OUTPUT_DTYPE = tl.float16
    elif input_feat.dtype == torch.bfloat16:
        OUTPUT_DTYPE = tl.bfloat16
    else:
        OUTPUT_DTYPE = tl.float32
    
    output = torch.empty((B, C_out, H, W), dtype=input_feat.dtype, device=input_feat.device)
    
    BLOCK_HW = 64
    BLOCK_CI = 64
    
    total_hw = B * H * W
    grid = (triton.cdiv(total_hw, BLOCK_HW),)
    
    # Get strides
    si = input_feat.stride()
    so = output.stride()
    
    conv1x1_kernel[grid](
        input_ptr=input_feat, weight_ptr=weight, bias_ptr=bias, output_ptr=output,
        B=B, C_IN=C_in, H=H, W=W,
        C_OUT=C_out,
        stride_ib=si[0], stride_ic=si[1], stride_ih=si[2], stride_iw=si[3],
        stride_ob=so[0], stride_oc=so[1], stride_oh=so[2], stride_ow=so[3],
        BLOCK_HW=BLOCK_HW, BLOCK_CI=BLOCK_CI,
        OUTPUT_DTYPE=OUTPUT_DTYPE,
    )
    
    return output

# Shared dispatch wrapper with all route branches
@torch.fx.wrap
def dispatch_wrapper(*args):
    route = args[-1]
    if route == "add_dropout2d":
        return triton_add(args[0], args[1])
    elif route == "conv2d_1x1":
        return triton_conv2d_1x1(args[0], args[1], args[2])
    else:
        raise ValueError(f"Unknown route: {route}")

def replacement_func():
    return dispatch_wrapper