import torch
import triton
import triton.language as tl


def pattern(in_3, in_1, in_0):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.permute(0, 3, 1, 2)
    return tmp_3


def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)


@torch.fx.wrap
def optimized_linear_permute(in_3, in_1, in_0):
    # Use torch.linalg.matmul for a simple but optimized linear operation
    # This is more efficient than the separate linear + permute sequence
    batch_size, seq_len1, seq_len2, hidden_dim = in_3.shape
    output_dim = in_1.shape[0]
    
    # Reshape input to [batch_size * seq_len1 * seq_len2, hidden_dim]
    x_flat = in_3.reshape(-1, hidden_dim)
    
    # Linear operation: x_flat @ in_1.T + in_0 (broadcasting)
    linear_result = torch.addmm(in_0, x_flat, in_1.T)
    
    # Reshape and permute to the desired output format
    # First reshape to [batch_size, seq_len1, seq_len2, output_dim]
    linear_reshaped = linear_result.reshape(batch_size, seq_len1, seq_len2, output_dim)
    
    # Permute dimensions from [batch, seq_len1, seq_len2, output_dim] to [batch, output_dim, seq_len1, seq_len2]
    result = linear_reshaped.permute(0, 3, 1, 2)
    
    return result


def replacement_func():
    return optimized_linear_permute