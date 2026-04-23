import torch
import triton
import triton.language as tl


def pattern(in_3: torch.Tensor, in_1: torch.Tensor, in_0: torch.Tensor):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_4 = linear.view(1, 12, 199, 2, 4)
    tmp_5 = tmp_4.sum(-1, keepdim=False)
    return tmp_5

def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

@triton.jit
def fused_linear_sum_kernel(input_ptr, weight_ptr, bias_ptr, output_ptr, batch_size, num_heads, seq_len, input_dim, output_dim_groups, output_dim_group_size, sum_bias0, sum_bias1):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    seq = tl.program_id(2)
    
    # Compute group 0 sum (indices 0-3)
    sum0 = 0.0
    for k in range(input_dim):
        input_val = tl.load(input_ptr + batch * num_heads * seq_len * input_dim + head * seq_len * input_dim + seq * input_dim + k)
        for j in range(output_dim_group_size):
            weight_val = tl.load(weight_ptr + j * input_dim + k)
            sum0 += input_val * weight_val
    sum0 += sum_bias0
    
    # Compute group 1 sum (indices 4-7)
    sum1 = 0.0
    for k in range(input_dim):
        input_val = tl.load(input_ptr + batch * num_heads * seq_len * input_dim + head * seq_len * input_dim + seq * input_dim + k)
        for j in range(output_dim_group_size, 2 * output_dim_group_size):
            weight_val = tl.load(weight_ptr + j * input_dim + k)
            sum1 += input_val * weight_val
    sum1 += sum_bias1
    
    # Store results
    output_offset = batch * num_heads * seq_len * output_dim_groups + head * seq_len * output_dim_groups + seq * output_dim_groups
    tl.store(output_ptr + output_offset, sum0)
    tl.store(output_ptr + output_offset + 1, sum1)

@torch.fx.wrap
def fused_linear_sum(in_3, in_1, in_0):
    import triton
    @triton.jit
    def sum_bias_kernel(bias_ptr, offset, sum_ptr):
        sum_val = 0.0
        for i in range(4):
            sum_val += tl.load(bias_ptr + offset + i)
        tl.store(sum_ptr, sum_val)

    sum_bias0_tensor = torch.empty(1, dtype=in_0.dtype, device=in_0.device)
    sum_bias1_tensor = torch.empty(1, dtype=in_0.dtype, device=in_0.device)
    grid = (1,)
    sum_bias_kernel[grid](in_0, 0, sum_bias0_tensor)
    sum_bias_kernel[grid](in_0, 4, sum_bias1_tensor)
    sum_bias0 = sum_bias0_tensor.item()
    sum_bias1 = sum_bias1_tensor.item()
    
    batch_size, num_heads, seq_len, input_dim = in_3.shape
    output_shape = (batch_size, num_heads, seq_len, 2)
    output = torch.empty(output_shape, dtype=in_3.dtype, device=in_3.device)
    
    grid = (batch_size, num_heads, seq_len)
    fused_linear_sum_kernel[grid](
        in_3, 
        in_1, 
        in_0,
        output,
        batch_size,
        num_heads,
        seq_len,
        input_dim,
        2,  
        4,  
        sum_bias0,
        sum_bias1
    )
    
    return output

def replacement_func():
    return fused_linear_sum