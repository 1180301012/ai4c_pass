import torch
import triton
import triton.language as tl

# Pattern matching: match sigmoid + view + broadcast multiply + contiguous
def pattern(conv2d_result, in_2):
    tmp_3 = torch.sigmoid(conv2d_result)
    tmp_4 = tmp_3.view(1, -1, 1, 1)
    tmp_5 = in_2 * tmp_4
    tmp_6 = tmp_5.contiguous()
    return (tmp_6,)

def replacement_args(conv2d_result, in_2):
    return (conv2d_result, in_2)


@triton.jit
def fused_sigmoid_mul_kernel(
    sigmoid_src_ptr,
    feature_ptr,
    output_ptr,
    total_elements,
    C,
    HW,
    stride_sigmoid_c,
    stride_feature_c, stride_feature_h, stride_feature_w,
    stride_output_c, stride_output_h, stride_output_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    c_idx = offsets // HW
    hw_idx = offsets % HW
    
    # W dimension stride for contiguous row
    W_val = stride_feature_w
    h_idx = hw_idx // W_val
    w_idx = hw_idx % W_val
    
    # Load sigmoid source value, cast to float32 for computation
    sigmoid_ptr = sigmoid_src_ptr + c_idx * stride_sigmoid_c
    sigmoid_val = tl.load(sigmoid_ptr, mask=mask, other=0.0).to(tl.float32)
    sigmoid_val = tl.sigmoid(sigmoid_val)
    
    # Load feature using strides, cast to float32
    f_ptrs = feature_ptr + c_idx * stride_feature_c + h_idx * stride_feature_h + w_idx * stride_feature_w
    f_vals = tl.load(f_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    # Compute and store
    out_vals = f_vals * sigmoid_val
    o_ptrs = output_ptr + c_idx * stride_output_c + h_idx * stride_output_h + w_idx * stride_output_w
    tl.store(o_ptrs, out_vals, mask=mask)


@triton.jit
def fused_sigmoid_mul_kernel_contiguous(
    sigmoid_src_ptr,
    feature_ptr,
    output_ptr,
    total_elements,
    C,
    HW,
    stride_sigmoid_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    c_idx = offsets // HW
    
    # Load sigmoid source value, cast to float32 for computation
    sigmoid_val = tl.load(sigmoid_src_ptr + c_idx * stride_sigmoid_c, mask=mask, other=0.0).to(tl.float32)
    sigmoid_val = tl.sigmoid(sigmoid_val)
    
    # Load feature (contiguous, offsets work directly), cast to float32
    f_vals = tl.load(feature_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute and store
    out_vals = f_vals * sigmoid_val
    tl.store(output_ptr + offsets, out_vals, mask=mask)


@torch.fx.wrap
def fused_sigmoid_mul(conv2d_result, feature):
    C = conv2d_result.shape[1]
    _, _, H, W = feature.shape
    HW = H * W
    total_elements = C * H * W
    
    output = torch.empty_like(feature)
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    stride_sigmoid_c = conv2d_result.stride()[1]
    
    if feature.is_contiguous():
        fused_sigmoid_mul_kernel_contiguous[(num_programs,)](
            sigmoid_src_ptr=conv2d_result,
            feature_ptr=feature,
            output_ptr=output,
            total_elements=total_elements,
            C=C,
            HW=HW,
            stride_sigmoid_c=stride_sigmoid_c,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        fused_sigmoid_mul_kernel[(num_programs,)](
            sigmoid_src_ptr=conv2d_result,
            feature_ptr=feature,
            output_ptr=output,
            total_elements=total_elements,
            C=C,
            HW=HW,
            stride_sigmoid_c=stride_sigmoid_c,
            stride_feature_c=feature.stride()[1],
            stride_feature_h=feature.stride()[2],
            stride_feature_w=feature.stride()[3],
            stride_output_c=output.stride()[1],
            stride_output_h=output.stride()[2],
            stride_output_w=output.stride()[3],
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return output


def replacement_func():
    return fused_sigmoid_mul