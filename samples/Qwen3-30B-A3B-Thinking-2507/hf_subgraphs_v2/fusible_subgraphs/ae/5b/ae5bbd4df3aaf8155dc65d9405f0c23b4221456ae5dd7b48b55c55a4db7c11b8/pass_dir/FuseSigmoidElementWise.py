import torch
import triton
import triton.language as tl


def pattern(tmp_5: torch.Tensor, in_2: torch.Tensor):
    tmp_6 = torch.sigmoid(tmp_5)
    chunk = tmp_6.chunk(2, dim=-1)
    tmp_8 = chunk[0]
    tmp_9 = chunk[1]
    tmp_10 = tmp_9 * in_2
    tmp_11 = tmp_10 - 1.0
    tmp_12 = tmp_8 * tmp_11
    tmp_13 = tmp_12 + 2.0
    return tmp_13

def replacement_args(tmp_5, in_2):
    return (tmp_5, in_2)

@triton.jit
def fused_sigmoid_eltwise_kernel(input_ptr, in2_ptr, output_ptr, batch_size, num_heads, seq_len):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    seq = tl.program_id(2)
    
    # Load v0 (index 0) and v1 (index 1) from input
    v0 = tl.load(input_ptr + batch * num_heads * seq_len * 2 + head * seq_len * 2 + seq * 2)
    v1 = tl.load(input_ptr + batch * num_heads * seq_len * 2 + head * seq_len * 2 + seq * 2 + 1)
    
    # Compute sigmoid(v1)
    v1_fp32 = v1.to(tl.float32)
    sigmoid_val = 1.0 / (1.0 + tl.exp(-v1_fp32))
    
    # Load in2 value (broadcasted to [1, num_heads, 1, 1])
    in2_val = tl.load(in2_ptr + 0 * num_heads * 1 * 1 + head * 1 * 1 + 0 * 1 + 0)
    
    # Compute final result: v0 * (sigmoid_val * in2_val - 1.0) + 2.0
    result = v0 * (sigmoid_val * in2_val - 1.0) + 2.0
    
    # Store result at (batch, head, seq, 0)
    output_offset = batch * num_heads * seq_len + head * seq_len + seq
    tl.store(output_ptr + output_offset, result)

@torch.fx.wrap
def fused_sigmoid_eltwise(tmp_5, in_2):
    batch_size, num_heads, seq_len, _ = tmp_5.shape
    output_shape = (batch_size, num_heads, seq_len, 1)
    output = torch.empty(output_shape, dtype=tmp_5.dtype, device=tmp_5.device)
    
    grid = (batch_size, num_heads, seq_len)
    fused_sigmoid_eltwise_kernel[grid](
        tmp_5,
        in_2,
        output,
        batch_size,
        num_heads,
        seq_len
    )
    
    return output

def replacement_func():
    return fused_sigmoid_eltwise