import torch
import triton
import triton.language as tl

@triton.jit
def fused_additions_kernel(
    input1_ptr,           # in_2 (attention_scores)
    input2_ptr,           # scaled_sigmoid_output
    input3_ptr,           # in_3.unsqueeze(1).unsqueeze(0) 
    input4_ptr,           # in_3.unsqueeze(1).unsqueeze(0)
    out_ptr,
    input1_shape,
    input2_shape,
    input3_shape,
    input4_shape,
    BLOCK_SIZE: tl.constexpr,
):
    # This kernel fuses multiple addition operations with broadcasting
    # Steps: in_2 + scaled_sigmoid.unsqueeze(0) + mask.unsqueeze(0) + mask.unsqueeze(0)
    # Note: The second mask addition can be optimized by just multiplying by 2
    
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input1_shape[0] * input1_shape[1] * input1_shape[2] * input1_shape[3]
    
    # Load tensors - handle broadcasting
    # For simplicity, we'll process as 1D with broadcasting handled in kernel
    input1 = tl.load(input1_ptr + offsets, mask=mask, other=0.0)
    
    # Handle broadcasting for scaled_sigmoid (unsqueeze(0))
    # This should have shape [1, heads, seq_len, seq_len] but needs broadcasting
    input2_offset = (offsets // (input1_shape[1] * input1_shape[2] * input1_shape[3])) * input2_shape[1] * input2_shape[2] * input2_shape[3] + (offsets % (input2_shape[1] * input2_shape[2] * input2_shape[3]))
    input2 = tl.load(input2_ptr + input2_offset, mask=input2_offset < input2_shape[0] * input2_shape[1] * input2_shape[2] * input2_shape[3], other=0.0)
    
    # Handle double mask addition (can be optimized to 2 * mask)
    # Instead of adding mask twice, we can multiply by 2
    input3_offset = (offsets // (input1_shape[1] * input1_shape[2] * input1_shape[3])) * input3_shape[1] * input3_shape[2] * input3_shape[3] + (offsets % (input3_shape[1] * input3_shape[2] * input3_shape[3]))
    mask_val = tl.load(input3_ptr + input3_offset, mask=input3_offset < input3_shape[0] * input3_shape[1] * input3_shape[2] * input3_shape[3], other=0.0)
    
    # Compute: input1 + input2 + 2 * mask_val
    result = input1 + input2 + 2.0 * mask_val
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_additions_with_broadcast(attention_scores, scaled_sigmoid, mask):
    # Get input shapes
    att_shape = attention_scores.shape
    sigmoid_shape = scaled_sigmoid.shape
    mask_shape = mask.shape
    
    # Calculate output shape (same as attention_scores)
    output_shape = att_shape
    numel = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]
    
    BLOCK_SIZE = 1024
    num_programs = (numel + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(attention_scores)
    
    fused_additions_kernel[(num_programs,)](
        input1_ptr=attention_scores,
        input2_ptr=scaled_sigmoid,
        input3_ptr=mask.unsqueeze(0).unsqueeze(0),  # This gives us the double mask addition
        input4_ptr=mask,
        out_ptr=output,
        input1_shape=att_shape,
        input2_shape=sigmoid_shape,
        input3_shape=mask.unsqueeze(0).unsqueeze(0).shape,
        input4_shape=mask.shape,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def pattern(attention_scores, scaled_sigmoid, mask, unsqueeze1, unsqueeze2, sum1, sum2, sum3, view_result):
    # Pattern matching the addition sequence:
    # tmp_11 = tmp_10.unsqueeze(0)
    # tmp_12 = in_2 + tmp_11
    # tmp_13 = tmp_12.view(1, 64, 12, 64, 64)
    # tmp_14 = in_3.unsqueeze(1)
    # tmp_15 = tmp_14.unsqueeze(0)
    # tmp_16 = tmp_13 + tmp_15
    # tmp_17 = in_3.unsqueeze(1)
    # tmp_18 = tmp_17.unsqueeze(0)
    # tmp_19 = tmp_16 + tmp_18
    # tmp_20 = tmp_19.view(-1, 12, 64, 64)
    
    # We can fuse: in_2 + scaled_sigmoid.unsqueeze(0) + mask.unsqueeze(0) + mask.unsqueeze(0)
    # And then optimize to: in_2 + scaled_sigmoid.unsqueeze(0) + 2 * mask.unsqueeze(0)
    tmp_11 = scaled_sigmoid.unsqueeze(0)
    tmp_12 = attention_scores + tmp_11
    tmp_14 = mask.unsqueeze(1)
    tmp_15 = tmp_14.unsqueeze(0)
    tmp_16 = tmp_12 + tmp_15
    tmp_17 = mask.unsqueeze(1)
    tmp_18 = tmp_17.unsqueeze(0)
    tmp_19 = tmp_16 + tmp_18
    return tmp_19

def replacement_args(attention_scores, scaled_sigmoid, mask):
    return (attention_scores, scaled_sigmoid, mask)

def replacement_func():
    return fused_additions_with_broadcast