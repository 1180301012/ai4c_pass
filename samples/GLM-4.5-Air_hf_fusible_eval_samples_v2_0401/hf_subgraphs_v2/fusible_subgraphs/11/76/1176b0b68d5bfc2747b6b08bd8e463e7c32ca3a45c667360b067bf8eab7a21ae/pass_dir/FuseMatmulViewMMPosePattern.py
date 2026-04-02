import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    matmul = torch.matmul(in_1, in_0)
    tmp_1 = matmul.view(shape=None)  # Shape will be determined by the specific model
    return tmp_1

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def _fused_matmul_view_kernel(
    a_ptr, b_ptr, out_ptr,
    batch_size, heads, seq_len, dim,
    block_size_m: tl.constexpr, block_size_n: tl.constexpr, block_size_k: tl.constexpr
):
    pid = tl.program_id(0)
    batch_id = pid // heads
    head_id = pid % heads
    
    # Each program handles one head in a batch
    # Output from matmul: [batch, heads, dim, seq_len] @ [batch, heads, seq_len, 1] = [batch, heads, dim, 1]
    # Then viewed to [batch, dim, 1, 1]
    
    # For head_id, batch_id, we compute output[batch_id, head_id, :, :] = a[batch_id, head_id, :, :] @ b[batch_id, head_id, :, :]
    
    a_base = (batch_id * heads + head_id) * dim * seq_len
    b_base = (batch_id * heads + head_id) * seq_len * 1
    
    # Each thread handles one element in the output [dim, 1]
    m = tl.program_id(1) * block_size_m + tl.arange(0, block_size_m)
    
    m_mask = m < dim
    
    accum = tl.zeros((block_size_m, 1), dtype=tl.float32)
    
    # Matrix multiplication: A[dim, seq_len] @ B[seq_len, 1] = [dim, 1]
    for k in range(0, seq_len, block_size_k):
        if k + block_size_k > seq_len:
            continue
            
        a_offset = a_base + m[:, None] * seq_len + k
        b_offset = b_base + k * 1
        
        a_data = tl.load(a_ptr + a_offset, mask=m_mask[:, None], other=0.0).to(tl.float32)
        b_data = tl.load(b_ptr + b_offset, mask=tl.arange(0, block_size_k)[:, None] < seq_len, other=0.0).to(tl.float32)
        
        # Compute dot product: A[dim, block_size_k] @ B[block_size_k, 1] -> [dim, 1]
        accum += tl.sum(a_data * b_data[:, None], axis=1)
    
    # Store result in [dim, 1] shape
    out_offset = (batch_id * heads + head_id) * dim * 1 + m[:, None] * 1
    tl.store(out_ptr + out_offset, accum, mask=m_mask[:, None])

@torch.fx.wrap
def fused_matmul_view(in_0, in_1):
    batch_size = in_0.shape[0]
    heads = in_0.shape[1]
    seq_len = in_0.shape[2]  # This is the K dimension from in_0
    dim = in_1.shape[2]      # This is the feature_dim from in_1
    
    # Original output from matmul: [batch_size, heads, dim, 1]
    # Then it gets viewed to [batch_size, dim, 1, 1] (removing the singleton head dimension)
    matmul_output = torch.empty((batch_size, heads, dim, 1), dtype=in_0.dtype, device=in_0.device)
    
    # Calculate grid
    grid_size_m = (dim + 31) // 32
    grid = (batch_size * heads, grid_size_m, 1)
    
    _fused_matmul_view_kernel[grid](
        in_1, in_0, matmul_output,
        batch_size, heads, seq_len, dim,
        32, 1, 32
    )
    
    # Apply the view operation: [batch_size, heads, dim, 1] -> [batch_size, dim, 1, 1]
    # Remove the singleton head dimension
    return matmul_output.reshape(batch_size, dim, 1, 1)

def replacement_func():
    return fused_matmul_view