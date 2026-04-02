import torch
import triton
import triton.language as tl
import math

@triton.jit
def layernorm_kernel(
    input_ptr,
    output_ptr,
    weight_ptr,
    bias_ptr,
    n_tokens,
    n_features,
    eps: float,
    BLOCK_SIZE: tl.constexpr,
):
    """High-performance LayerNorm kernel"""
    # Each program负责处理一个token的所有features
    pid = tl.program_id(0)
    
    # 边界检查
    if pid >= n_tokens:
        return
    
    # 计算当前token的起始位置
    input_offset = pid * n_features
    output_offset = pid * n_features
    
    # 加载当前token的所有features到共享内存
    features = tl.load(input_ptr + input_offset + tl.arange(0, BLOCK_SIZE), 
                       mask=input_offset + tl.arange(0, BLOCK_SIZE) < n_features, 
                       other=0.0)
    
    # 使用更高效的方差计算方式 (避免重复计算tl.sum)
    sum_val = tl.sum(features)
    mean = sum_val / n_features
    
    features_centered = features - mean
    
    # 使用更高效的平方和计算，减少内存访问
    features_squared = features_centered * features_centered
    mean_sq = tl.sum(features_squared) / n_features
    
    # 使用rsqrt替代sqrt/divide for better performance
    rsqrt_val = tl.math.rsqrt(mean_sq + eps)
    normalized = features_centered * rsqrt_val
    
    # 加载weight和bias
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), 
                     mask=tl.arange(0, BLOCK_SIZE) < n_features, 
                     other=1.0)
    bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), 
                   mask=tl.arange(0, BLOCK_SIZE) < n_features, 
                   other=0.0)
    
    # 向量化计算: normalized * weight + bias
    output = normalized * weight + bias
    
    # 存储结果
    tl.store(output_ptr + output_offset + tl.arange(0, BLOCK_SIZE), 
             output, 
             mask=input_offset + tl.arange(0, BLOCK_SIZE) < n_features)

@torch.fx.wrap
def optimized_layernorm(input_tensor, weight, bias, eps=1e-12):
    """Optimized LayerNorm implementation using Triton"""
    # Handle 3D tensors: [batch_size, seq_len, features]
    batch_size, seq_len, n_features = input_tensor.shape
    
    # Reshape to 2D for processing: [batch_size * seq_len, features]
    input_2d = input_tensor.reshape(-1, n_features)
    
    # Output tensor
    output_2d = torch.empty_like(input_2d)
    
    # 选择block大小 - 根据feature维度调整
    BLOCK_SIZE = min(256, n_features)
    
    # 计算需要的program数量
    num_tokens = batch_size * seq_len
    
    # 启动kernel
    layernorm_kernel[(num_tokens,)](
        input_ptr=input_2d,
        output_ptr=output_2d,
        weight_ptr=weight,
        bias_ptr=bias,
        n_tokens=num_tokens,
        n_features=n_features,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to original 3D shape
    output = output_2d.reshape(batch_size, seq_len, n_features)
    return output

def pattern(x, normalized_shape, weight, bias, eps):
    """Match torch.nn.functional.layer_norm"""
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)

def replacement_args(x, normalized_shape, weight, bias, eps):
    """Extract arguments for replacement function"""
    return (x, weight, bias, eps)  # normalized_shape is inferred from input

def replacement_func():
    """Return the optimized LayerNorm function"""
    return optimized_layernorm