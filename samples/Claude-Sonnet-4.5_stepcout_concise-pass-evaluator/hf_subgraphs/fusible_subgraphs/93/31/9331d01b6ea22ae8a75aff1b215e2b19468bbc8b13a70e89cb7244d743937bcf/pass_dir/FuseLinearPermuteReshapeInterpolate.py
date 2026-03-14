import torch
import triton
import triton.language as tl

def pattern(in_3, tmp_1, tmp_0):
    """Pattern: linear -> permute -> reshape -> interpolate"""
    tmp_2 = torch.nn.functional.linear(in_3, tmp_1, tmp_0)
    tmp_3 = tmp_2.permute(0, 2, 1)
    tmp_4 = tmp_3.reshape(tmp_3.shape[0], -1, 64, 64)
    tmp_5 = torch.nn.functional.interpolate(tmp_4, size=(128, 128), mode='bilinear', align_corners=False)
    return tmp_5

def replacement_args(in_3, tmp_1, tmp_0):
    return (in_3, tmp_1, tmp_0)

@triton.jit
def fused_linear_permute_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    seq_len,
    in_features,
    out_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused linear + permute kernel
    Input: [B, S, I] -> Linear -> [B, S, O] -> Permute -> [B, O, S]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Offsets for output [B, O, S]
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Perform matrix multiplication in chunks
    for k in range(0, tl.cdiv(in_features, BLOCK_SIZE_K)):
        offs_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        
        # Load input [B, S, I]
        input_mask = (offs_m[:, None] < seq_len) & (offs_k[None, :] < in_features)
        input_ptrs = input_ptr + pid_b * seq_len * in_features + offs_m[:, None] * in_features + offs_k[None, :]
        input_vals = tl.load(input_ptrs, mask=input_mask, other=0.0)
        
        # Load weight [O, I] (transposed for computation)
        weight_mask = (offs_n[:, None] < out_features) & (offs_k[None, :] < in_features)
        weight_ptrs = weight_ptr + offs_n[:, None] * in_features + offs_k[None, :]
        weight_vals = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(input_vals, tl.trans(weight_vals))
    
    # Add bias
    if bias_ptr is not None:
        bias_mask = offs_n < out_features
        bias_vals = tl.load(bias_ptr + offs_n, mask=bias_mask, other=0.0)
        acc += bias_vals[None, :]
    
    # Store to output [B, O, S] (permuted layout)
    output_mask = (offs_m[:, None] < seq_len) & (offs_n[None, :] < out_features)
    output_ptrs = output_ptr + pid_b * out_features * seq_len + offs_n[None, :] * seq_len + offs_m[:, None]
    tl.store(output_ptrs, acc, mask=output_mask)

@torch.fx.wrap
def fused_linear_permute_reshape_interpolate(input_tensor, weight, bias):
    """
    Fused implementation of linear -> permute -> reshape -> interpolate
    """
    batch_size, seq_len, in_features = input_tensor.shape
    out_features = weight.shape[0]
    
    # First, do linear + permute with Triton
    linear_permute_output = torch.empty(batch_size, out_features, seq_len, 
                                       dtype=input_tensor.dtype, device=input_tensor.device)
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    grid = (triton.cdiv(seq_len, BLOCK_SIZE_M), 
            triton.cdiv(out_features, BLOCK_SIZE_N),
            batch_size)
    
    bias_ptr = bias if bias is not None else None
    
    fused_linear_permute_kernel[grid](
        input_tensor,
        weight,
        bias_ptr,
        linear_permute_output,
        batch_size,
        seq_len,
        in_features,
        out_features,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    # Reshape: [B, O, S] -> [B, O//64//64, 64, 64]
    # Since O = num_channels and S = H*W, we need to figure out the spatial dims
    # For the pattern, seq_len should be 64*64 = 4096, so we reshape to [B, C, 64, 64]
    spatial_dim = int(seq_len ** 0.5)
    num_channels = out_features
    reshaped = linear_permute_output.reshape(batch_size, num_channels, spatial_dim, spatial_dim)
    
    # Interpolate to 128x128
    result = torch.nn.functional.interpolate(reshaped, size=(128, 128), mode='bilinear', align_corners=False)
    
    return result

def replacement_func():
    return fused_linear_permute_reshape_interpolate