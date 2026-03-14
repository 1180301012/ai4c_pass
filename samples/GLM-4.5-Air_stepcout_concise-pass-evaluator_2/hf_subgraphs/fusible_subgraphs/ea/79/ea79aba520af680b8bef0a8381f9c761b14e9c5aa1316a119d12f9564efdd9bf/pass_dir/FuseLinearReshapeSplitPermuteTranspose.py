import torch
import triton
import triton.language as tl
import math

def linear(x, weight, bias):
    return torch.nn.functional.linear(x, weight, bias)

def reshape(x, shape):
    return torch.reshape(x, shape)

def split(x, sizes, dim):
    return torch.split(x, sizes, dim=dim)

def permute(x, dims):
    return torch.permute(x, dims)

def transpose(x, dim0, dim1):
    return torch.transpose(x, dim0, dim1)

def to_device(x, device):
    return x.to(device)

def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = linear(in_3, tmp_2, tmp_1)
    tmp_4 = reshape(tmp_3, (-1, 49, 8, -1))
    tmp_5 = split(tmp_4, [32, 32, 128], dim=3)
    tmp_6 = tmp_5[0]
    tmp_7 = tmp_5[1]
    tmp_8 = tmp_5[2]
    tmp_9 = permute(tmp_6, (0, 2, 1, 3))
    tmp_10 = permute(tmp_7, (0, 2, 1, 3))
    tmp_11 = permute(tmp_8, (0, 2, 1, 3))
    tmp_12 = to_device(tmp_0, device(type='cuda', index=0))
    tmp_13 = transpose(tmp_10, -2, -1)
    return (tmp_9, tmp_12, tmp_13, tmp_11)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)



@torch.fx.wrap
def fused_kernel(in_0, in_1, in_2, in_3):
    # Get tensor shapes and sizes
    batch_size = in_3.shape[0]
    seq_len = in_3.shape[1] 
    feat_in = in_3.shape[2]
    feat_out = in_1.shape[0]  # 1536
    head_dim = 8  # From reshape(..., 8, -1)
    split_sizes = [32, 32, 128]
    
    # Create output tensors
    out1_shape = (batch_size, head_dim, seq_len, split_sizes[0])
    out2_shape = (batch_size, head_dim, split_sizes[1], seq_len)
    out3_shape = (batch_size, head_dim, seq_len, split_sizes[2])
    
    out1 = torch.empty(out1_shape, dtype=torch.float32, device='cuda')
    out2 = torch.empty(out2_shape, dtype=torch.float32, device='cuda')
    out3 = torch.empty(out3_shape, dtype=torch.float32, device='cuda')
    
    # Use efficient linear computation
    tmp_3 = torch.nn.functional.linear(in_3, in_2, in_1)
    
    # Fused reshape, split, and permute operations to avoid intermediate allocations
    # Reshape to [batch, seq_len, head_dim, split_total] 
    tmp_4 = tmp_3.reshape(batch_size, seq_len, head_dim, -1)  # [batch, seq_len, 8, 192]
    
    # Efficient split and permutation in one step using tensor slicing
    # Split along the last dimension and permute each slice
    # For each split, we want [batch, head_dim, seq_len, split_size]
    
    # Split 0: [32]
    split0_slice = tmp_4[..., :split_sizes[0]]
    out1 = split0_slice.permute(0, 2, 1, 3)  # [batch, 8, seq_len, 32]
    
    # Split 1: [32] -> [batch, 8, seq_len, 32] -> [batch, 8, 32, seq_len]
    split1_slice = tmp_4[..., split_sizes[0]:split_sizes[0]+split_sizes[1]]
    tmp_10 = split1_slice.permute(0, 2, 1, 3)  # [batch, 8, seq_len, 32]
    out2 = tmp_10.transpose(-2, -1)  # [batch, 8, 32, seq_len]
    
    # Split 2: [128]
    split2_slice = tmp_4[..., split_sizes[0]+split_sizes[1]:]
    out3 = split2_slice.permute(0, 2, 1, 3)  # [batch, 8, seq_len, 128]
    
    # Device transfer for in_0
    tmp_12 = in_0.to(device='cuda')
    
    return (out1, tmp_12, out2, out3)

def replacement_func():
    return fused_kernel