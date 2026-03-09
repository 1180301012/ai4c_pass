import torch
import triton
import triton.language as tl

# Let's try matching linear operation specifically
def pattern(x, weight, bias=None):
    return x @ weight.T + (bias if bias is not None else 0)

def replacement_args(x, weight, bias=None):
    return (x, weight, bias)

def replacement_func():
    # Simple Triton linear kernel for demonstration
    import triton
    import triton.language as tl
    
    @triton.jit
    def linear_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        n_batch,
        n_seq,
        d_in,
        d_out,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        row_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        col_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        
        row_mask = row_offset < n_batch * n_seq
        col_mask = col_offset < d_out
        
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        for k in range(0, d_in, BLOCK_SIZE_K):
            x_ptrs = x_ptr + (row_offset[:, None] * d_in + k + tl.arange(0, BLOCK_SIZE_K)[None, :])
            weight_ptrs = weight_ptr + (k * d_out + col_offset[None, :])
            
            x_data = tl.load(x_ptrs, mask=(row_mask[:, None] & (k + tl.arange(0, BLOCK_SIZE_K)[None, :]) < d_in), other=0.0)
            weight_data = tl.load(weight_ptrs, mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < d_in) & col_mask[None, :], other=0.0)
            
            acc += tl.dot(x_data, weight_data)
        
        if bias_ptr is not None:
            bias_data = tl.load(bias_ptr + col_offset[None, :], mask=col_mask[None, :], other=0.0)
            acc += bias_data
        
        out_ptrs = out_ptr + (row_offset[:, None] * d_out + col_offset[None, :])
        tl.store(out_ptrs, acc, mask=row_mask[:, None] & col_mask[None, :])
    
    @torch.fx.wrap  
    def linear_op(x, weight, bias=None):
        n_batch, n_seq, d_in = x.shape
        d_out = weight.shape[0]
        
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
        
        m_blocks = (n_batch * n_seq + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        n_blocks = (d_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        
        output = torch.empty(n_batch, n_seq, d_out, device=x.device, dtype=x.dtype)
        
        linear_kernel[(m_blocks, n_blocks)](
            x, weight, bias if bias is not None else torch.empty(d_out, device=x.device, dtype=x.dtype),
            output, n_batch * n_seq, d_out, d_in, d_out,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
        )
        
        return output
    
    return linear_op