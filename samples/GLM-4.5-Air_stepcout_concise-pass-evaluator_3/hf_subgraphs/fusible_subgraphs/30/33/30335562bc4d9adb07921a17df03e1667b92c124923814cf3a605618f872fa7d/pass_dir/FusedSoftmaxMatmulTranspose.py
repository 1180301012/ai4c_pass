import torch
import triton
import triton.language as tl


# Pattern matching function - matches the computation pattern:
# This must start from the graph inputs (in_0, in_1) and produce the final output
# All input arguments must be used (no dead code)
def pattern(in_0, in_1):
    # Step 1: Scale the input
    tmp_0 = 0.0625 * in_0
    # Step 2: Apply softmax on last dimension
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    # Step 3: Matrix multiplication  
    tmp_2 = torch.matmul(tmp_1, in_1)
    # Step 4: Permute to get [B, N, H] from [B, H, N]
    tmp_3 = tmp_2.permute(0, 2, 1)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# Optimized Triton kernel that fuses:
# 1. Scalar multiplication (0.0625 * in_0)
# 2. Softmax on last dimension
# 3. Matmul with in_1
# 4. Transpose (permute)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_N': 256}, num_stages=3, num_warps=8),
    ],
    key=['B', 'M', 'K', 'N'],
)
@triton.jit
def fused_softmax_matmul_transpose_kernel(
    # Input pointers
    in_0_ptr, in_1_ptr,
    # Output pointer
    out_ptr,
    # Shapes
    B, M, K, N,
    # Strides
    stride_in_0_b, stride_in_0_m, stride_in_0_k,
    stride_in_1_b, stride_in_1_n, stride_in_1_k,
    stride_out_b, stride_out_n, stride_out_m,
    # Block size for output dimension
    BLOCK_N: tl.constexpr,
):
    """
    Fused kernel that computes:
    1. out = 0.0625 * in_0  [B, M, K]
    2. softmax(out, dim=-1)  [B, M, K]
    3. matmul(softmax_result, in_1)  [B, M, N]
    4. permute to [B, N, M]
    """
    # Get batch and sequence index
    pid = tl.program_id(0)
    batch_idx = pid // M
    seq_idx = pid % M
    
    if batch_idx >= B:
        return
    
    in_0_base = batch_idx * stride_in_0_b + seq_idx * stride_in_0_m
    in_1_base = batch_idx * stride_in_1_b
    out_base = batch_idx * stride_out_b
    
    # First pass: find max (from scaled values for numerical stability)
    max_val = float('-inf')
    for k in range(K):
        ptr = in_0_base + k * stride_in_0_k
        val = tl.load(ptr).to(tl.float32)
        scaled_val = val * 0.0625
        max_val = tl.max(max_val, scaled_val)
    
    # Second pass: compute exp with numerical stability
    exp_sum = float(0)
    softmax_vals = []
    for k in range(K):
        ptr = in_0_base + k * stride_in_0_k
        val = tl.load(ptr).to(tl.float32)
        scaled_val = val * 0.0625
        exp_val = tl.exp(scaled_val - max_val)
        softmax_vals.append(exp_val)
        exp_sum = exp_sum + exp_val
    
    # Normalize softmax values
    softmax_vals = [v / exp_sum for v in softmax_vals]
    
    # Compute matmul for each output dimension block
    for n_block in range(0, N, BLOCK_N):
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        
        for k in range(K):
            sval = softmax_vals[k]
            
            n_offsets = n_block + tl.arange(0, BLOCK_N)
            mask = n_offsets < N
            
            in_1_ptrs = in_1_base + k * stride_in_1_k + n_offsets * stride_in_1_n
            in_1_vals = tl.load(in_1_ptrs, mask=mask, other=0.0).to(tl.float32)
            
            acc = acc + sval * in_1_vals
        
        n_offsets = n_block + tl.arange(0, BLOCK_N)
        mask = n_offsets < N
        
        out_ptrs = out_base + n_offsets * stride_out_n + seq_idx * stride_out_m
        tl.store(out_ptrs, acc, mask=mask)


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1):
    """Fused kernel wrapper for scale + softmax + matmul + transpose."""
    B, M, K = in_0.shape
    _, _, N = in_1.shape
    
    output = torch.empty((B, N, M), dtype=in_0.dtype, device=in_0.device)
    grid = (B * M,)
    
    fused_softmax_matmul_transpose_kernel[grid](
        in_0, in_1, output,
        B, M, K, N,
        in_0.stride(0), in_0.stride(1), in_0.stride(2),
        in_1.stride(0), in_1.stride(1), in_1.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
    )
    
    return output


def replacement_func():
    return fused_kernel_wrapper