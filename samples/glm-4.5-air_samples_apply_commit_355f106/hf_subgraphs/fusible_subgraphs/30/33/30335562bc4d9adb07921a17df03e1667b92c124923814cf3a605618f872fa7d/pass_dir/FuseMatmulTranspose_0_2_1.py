import torch
import triton
import triton.language as tl

# Pattern: matrix multiplication followed by transpose(0, 2, 1)
def pattern(*args):
    tmp_1 = args[0]  # First argument is the first input tensor
    in_1 = args[1]   # Second argument is the second input tensor
    tmp_2 = torch.matmul(tmp_1, in_1)
    tmp_3 = tmp_2.permute(0, 2, 1)
    return tmp_3

# Arguments for replacement - need both input tensors
def replacement_args(*args):
    return (args[0], args[1])  # Return first two arguments

# Optimized kernel that fuses matmul with transpose(0, 2, 1)
@triton.jit
def fused_matmul_transpose_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_batch,
    n_a_seq,      # seq_len from a (8192)
    n_a_feat,     # features from a (19)
    n_b_feat,     # features from b (must match n_a_feat)
    n_b_out,      # output dimension from b (256)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Matrix multiplication with fusion
    pid_m = tl.program_id(0)  # batch dimension
    pid_n = tl.program_id(1)  # output dimension (256)
    pid_k = tl.program_id(2)  # seq_len dimension (8192)
    
    # Calculate offsets for this work item
    batch_offset = pid_m * n_a_seq * n_b_out
    
    # Create offsets within matrices
    a_offset = pid_m * n_a_seq * n_a_feat
    b_offset = pid_m * n_b_feat * n_b_out
    out_offset = batch_offset + pid_n * n_a_seq + pid_k
    
    # Load matrices tiles
    a_ptrs = (a_offset + pid_k * n_a_feat + tl.arange(0, BLOCK_SIZE_K)).to(tl.int64)
    b_ptrs = (b_offset + tl.arange(0, BLOCK_SIZE_N)[:, None] + n_b_out * tl.arange(0, BLOCK_SIZE_K)[None, :]).to(tl.int64)
    a = tl.load(a_ptr + a_ptrs, mask=tl.arange(0, BLOCK_SIZE_K) < n_a_feat, other=0.0)
    b = tl.load(b_ptr + b_ptrs, mask=(tl.arange(0, BLOCK_SIZE_N)[:, None] < n_b_out) & (tl.arange(0, BLOCK_SIZE_K)[None, :] < n_b_feat), other=0.0)
    
    # Compute matrix multiplication
    acc = tl.dot(a, b)
    
    # For transposition: original output is [batch, seq_len, output_dim], we want [batch, output_dim, seq_len]
    # So we're computing element at [batch, output_dim, seq_len] directly
    tl.store(out_ptr + out_offset, acc)

@torch.fx.wrap
def fused_matmul_transpose(a, b):
    # Get tensor dimensions
    n_batch = a.shape[0]
    n_a_seq = a.shape[1]
    n_a_feat = a.shape[2]
    n_b_feat = b.shape[1]  # Must match n_a_feat for matmul
    n_b_out = b.shape[2]
    
    # Create output tensor with transposed shape: [batch, output_dim, seq_len]
    out_shape = (n_batch, n_b_out, n_a_seq)
    out = torch.empty(out_shape, dtype=a.dtype, device=a.device)
    
    # Block sizes for optimal GPU occupancy
    BLOCK_SIZE_M = 1      # Process one batch at a time
    BLOCK_SIZE_N = 64     # Block in output dimension
    BLOCK_SIZE_K = 32     # Block in seq dimension
    
    # Calculate grid dimensions
    grid = (
        n_batch,
        (n_b_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
        (n_a_seq + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    )
    
    # Launch kernel
    fused_matmul_transpose_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        out_ptr=out,
        n_batch=n_batch,
        n_a_seq=n_a_seq,
        n_a_feat=n_a_feat,
        n_b_feat=n_b_feat,
        n_b_out=n_b_out,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out

def replacement_func():
    return fused_matmul_transpose