import torch
import triton
import triton.language as tl

def pattern(x_3, weight, bias, x_2):
    # Pattern matching the full computation:
    # tmp_2 = torch.conv2d(x_3, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    # tmp_3 = torch.nn.functional.hardsigmoid(tmp_2, False)
    # tmp_4 = x_2 * tmp_3
    # tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
    # tmp_6 = tmp_5.flatten(1, -1)
    # tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    # return (tmp_7,)
    
    conv_result = torch.conv2d(x_3, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    hardsigmoid_result = torch.nn.functional.hardsigmoid(conv_result, False)
    mul_result = x_2 * hardsigmoid_result
    pool_result = torch.nn.functional.adaptive_avg_pool2d(mul_result, 1)
    flatten_result = pool_result.flatten(1, -1)
    final_result = torch.nn.functional.dropout(flatten_result, 0.0, False, False)
    return final_result

def replacement_args(x_3, weight, bias, x_2):
    return (x_3, weight, bias, x_2)

@torch.fx.wrap
def optimized_full_graph(x_3, weight, bias, x_2):
    """Optimized implementation of the full computation graph - using native PyTorch with optimizations"""
    if x_3.numel() == 0 or weight.numel() == 0 or bias.numel() == 0 or x_2.numel() == 0:
        # Return empty tensor with correct shape if inputs are empty
        N, C_out, _, _ = x_2.shape  # Use x_2 dimensions as reference
        return torch.empty((N, C_out), dtype=x_3.dtype, device=x_3.device)
    
    N, C_out, H2, W2 = x_2.shape
    C_in = weight.shape[1]
    
    # Compute the result using the original pattern but with some optimizations:
    # 1. Reshape tensors to avoid 1x1 convolution overhead when possible
    # 2. Use more efficient pooling operation
    # 3. Remove identity operations
    
    x_3_flat = x_3.reshape(N, C_in)  # [N, C_in, 1, 1] -> [N, C_in]
    weight_flat = weight.view(C_out, C_in)  # [C_out, C_in, 1, 1] -> [C_out, C_in]
    
    # Use einsum for efficient linear operation (equivalent to 1x1 conv)
    conv_result = torch.einsum('ni,oi->no', x_3_flat, weight_flat) + bias
    conv_result = conv_result.reshape(N, C_out, 1, 1)  # [N, C_out, 1, 1]
    
    # Manually implement hardsigmoid to avoid dependency on torch.nn.functional
    hardsigmoid_result = torch.clamp(conv_result + 3.0, min=0.0, max=1.0) / 6.0
    
    # Element-wise multiplication with broadcasting
    mul_result = x_2 * hardsigmoid_result
    
    # Replace adaptive_avg_pool2d(1) with direct mean computation for efficiency
    pool_result = mul_result.flatten(start_dim=2).mean(dim=-1)  # [N, C_out]
    
    # Dropout with p=0.0 is identity - skip it completely
    
    return pool_result

def replacement_func():
    return optimized_full_graph