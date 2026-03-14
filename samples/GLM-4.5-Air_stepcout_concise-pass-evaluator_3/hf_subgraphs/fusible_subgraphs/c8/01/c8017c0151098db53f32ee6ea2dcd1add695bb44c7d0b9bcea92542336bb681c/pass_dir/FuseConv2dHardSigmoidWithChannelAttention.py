import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardsigmoid(tmp_2, False)
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
    tmp_6 = tmp_5.flatten(1, -1)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    return (tmp_7,)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def conv2d_hardsigmoid_kernel(
    x_ptr, weight_ptr, bias_ptr,
    out_ptr, 
    N, C_out, H_out, W_out,
    C_in_kh_kw, kh, kw,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.cdiv(N * C_out * H_out * W_out, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * C_out * H_out * W_out
    
    # Reshape offsets to [N, C_out, H_out, W_out]
    n = offsets // (C_out * H_out * W_out)
    c = (offsets % (C_out * H_out * W_out)) // (H_out * W_out)
    h = (offsets % (H_out * W_out)) // W_out
    w = offsets % W_out
    
    # Load input (shape [N, C_in, 1, 1] for this SE module)
    x = tl.load(x_ptr + n * C_in_kh_kw + c * kh * kw + h * kw + w, mask=mask)
    
    # Load weight (shape [C_out, C_in, 1, 1])
    weight = tl.load(weight_ptr + c * C_in_kh_kw, mask=mask)
    
    # Load bias (shape [C_out])
    bias = tl.load(bias_ptr + c, mask=mask)
    
    # Conv2D operation (1x1 convolution)
    conv_out = weight * x + bias
    
    # HardSigmoid activation: max(0, min(1, x + 3)) / 6
    hardsigmoid_out = tl.where(conv_out > -3.0, 
                              tl.where(conv_out < 3.0, (conv_out + 3.0) / 6.0, 1.0), 
                              0.0)
    
    # Store hardsigmoid output (shape [N, C_out, 1, 1])
    tl.store(out_ptr + n * C_out + c, hardsigmoid_out, mask=mask)

@triton.jit
def elementwise_multiply_kernel(
    feature_map_ptr, se_output_ptr,
    output_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.cdiv(N * C * H * W, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * C * H * W
    
    # Load feature map and SE output
    feature_map = tl.load(feature_map_ptr + offsets, mask=mask)
    se_output = tl.load(se_output_ptr + offsets // (H * W), mask=mask)
    
    # Element-wise multiplication
    output = feature_map * se_output
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def avg_pool2d_flatten_kernel(
    input_ptr,
    output_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.cdiv(N * C, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * C
    
    # Load input (shape [N, C, H, W])
    offsets_HW = offsets % (H * W)
    offsets_C = (offsets // (H * W)) % C
    offsets_N = offsets // (C * H * W)
    
    input = tl.load(input_ptr + offsets_N * C * H * W + offsets_C * H * W + offsets_HW, mask=mask)
    
    # Global average pooling across spatial dimensions
    avg_pool = input / (H * W)
    
    # Store result (flattened: [N, C])
    tl.store(output_ptr + offsets, avg_pool, mask=mask)

@torch.fx.wrap
def channel_attention_conv_hardsigmoid(x, weight, bias, out_N, out_C, out_H, out_W, in_C):
    # Determine optimal grid size
    total_elements = out_N * out_C * out_H * out_W
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty((out_N, out_C, out_H, out_W), dtype=torch.float32, device=x.device)
    
    # Launch kernel for Conv2D + HardSigmoid fusion
    conv2d_hardsigmoid_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        N=out_N, C_out=out_C, H_out=out_H, W_out=out_W,
        C_in_kh_kw=in_C * 1 * 1, kh=1, kw=1,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

@torch.fx.wrap
def elementwise_multiply_se_attention(feature_map, se_output, N, C, H, W):
    total_elements = N * C * H * W
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty((N, C, H, W), dtype=torch.float32, device=feature_map.device)
    
    elementwise_multiply_kernel[(num_programs,)](
        feature_map_ptr=feature_map,
        se_output_ptr=se_output,
        output_ptr=out,
        N=N, C=C, H=H, W=W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

@torch.fx.wrap
def avg_pool2d_flatten_attention(input_tensor, N, C, H, W):
    total_elements = N * C
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty((N, C), dtype=torch.float32, device=input_tensor.device)
    
    avg_pool2d_flatten_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=out,
        N=N, C=C, H=H, W=W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    def optimized_forward(in_0, in_1, in_2, in_3):
        # Get input shapes
        in_1_shape = in_1.shape
        in_2_shape = in_2.shape
        in_3_shape = in_3.shape
        
        # Conv2D + HardSigmoid fusion for SE module
        se_output = channel_attention_conv_hardsigmoid(
            x=in_3, 
            weight=in_1, 
            bias=in_0,
            out_N=in_3_shape[0],
            out_C=in_1_shape[0],  # C_out
            out_H=in_3_shape[2],  #保持空间维度 (通常是1x1)
            out_W=in_3_shape[3],  #保持空间维度 (通常是1x1)
            in_C=in_1_shape[1]   # C_in
        )
        
        # Element-wise multiplication with feature map
        multiplied = elementwise_multiply_se_attention(
            feature_map=in_2,
            se_output=se_output,
            N=in_2_shape[0],
            C=in_2_shape[1],
            H=in_2_shape[2],
            W=in_2_shape[3]
        )
        
        # Global average pooling + flatten (optimized for 0.0 dropout)
        pooled_flattened = avg_pool2d_flatten_attention(
            input_tensor=multiplied,
            N=in_2_shape[0],
            C=in_2_shape[1],
            H=in_2_shape[2],
            W=in_2_shape[3]
        )
        
        # Return since dropout with 0.0 rate is identity operation
        return (pooled_flattened,)
    
    return optimized_forward