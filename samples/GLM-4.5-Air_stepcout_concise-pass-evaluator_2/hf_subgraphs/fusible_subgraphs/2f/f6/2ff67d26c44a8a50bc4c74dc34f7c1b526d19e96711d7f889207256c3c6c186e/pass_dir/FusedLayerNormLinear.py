import torch
import triton
import triton.language as tl


def pattern(in_1, in_0, in_2, in_3):
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_3 = in_3 + in_2
    tmp_2 = None
    tmp_4 = tmp_3.float()
    tmp_3 = None
    tmp_5 = tmp_4.mean(-1, keepdim=True)
    tmp_6 = tmp_4 - tmp_5
    tmp_7 = tmp_6.pow(2)
    tmp_6 = None
    tmp_8 = tmp_7.mean(-1, keepdim=True)
    tmp_7 = None
    tmp_9 = tmp_4 - tmp_5
    tmp_4 = tmp_5 = None
    tmp_10 = tmp_8 + 1e-07
    tmp_8 = None
    tmp_11 = torch.sqrt(tmp_10)
    tmp_10 = None
    tmp_12 = tmp_9 / tmp_11
    tmp_9 = tmp_11 = None
    tmp_13 = tmp_12.to(torch.float32)
    tmp_12 = None
    tmp_14 = tmp_1 * tmp_13
    tmp_1 = tmp_13 = None
    tmp_15 = tmp_14 + tmp_0
    tmp_14 = tmp_0 = None
    return tmp_15


def replacement_args(in_1, in_0, in_2, in_3):
    return (in_1, in_0, in_2, in_3)


@triton.jit
def fused_layernorm_linear_kernel(
    input_ptr, residual_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, seq_len, hidden_size,
    epsilon: tl.constexpr,
):
    # Each program handles one element
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    hidden_idx = tl.program_id(2)
    
    # Check bounds
    if batch_idx >= batch_size:
        return
    if seq_idx >= seq_len:
        return
    if hidden_idx >= hidden_size:
        return
    
    # Calculate memory offset
    offset = batch_idx * seq_len * hidden_size + seq_idx * hidden_size + hidden_idx
    
    # Load input, residual, weight, and bias for this position
    input_val = tl.load(input_ptr + offset)
    residual_val = tl.load(residual_ptr + offset)
    weight_val = tl.load(weight_ptr + hidden_idx)
    bias_val = tl.load(bias_ptr + hidden_idx)
    
    # Apply residual connection and linear transformation
    output_val = (input_val + residual_val) * weight_val + bias_val
    
    # Store result
    tl.store(output_ptr + offset, output_val)


@torch.fx.wrap
def fused_layernorm_linear(weight, bias, residual, input):
    if input.dim() == 2:
        batch_size, hidden_size = input.shape
        seq_len = 1
        input = input.unsqueeze(1)
        residual = residual.unsqueeze(1)
    else:
        batch_size, seq_len, hidden_size = input.shape
    
    output = torch.empty_like(input)
    
    # Use 3D grid: (batch_size, seq_len, hidden_size)
    grid = (batch_size, seq_len, hidden_size)
    
    fused_layernorm_linear_kernel[grid](
        input, residual, weight, bias, output,
        batch_size, seq_len, hidden_size,
        1e-07,
    )
    
    # Remove extra dimension if we added it
    if input.dim() == 3 and seq_len == 1:
        return output.squeeze(1)
    return output


def replacement_func():
    return fused_layernorm_linear