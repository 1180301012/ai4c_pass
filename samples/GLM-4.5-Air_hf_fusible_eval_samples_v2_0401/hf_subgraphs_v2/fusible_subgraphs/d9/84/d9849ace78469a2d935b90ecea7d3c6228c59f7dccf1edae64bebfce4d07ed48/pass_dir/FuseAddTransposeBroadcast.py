import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Element-wise addition with broadcasting (in_0: [128, 1] broadcasts to [1, 128, 19])
    in_1 += in_0
    in_2 = in_1
    # Transpose from [1, 128, 19] to [1, 19, 128]
    tmp_2 = in_2.transpose(1, 2)
    return (tmp_2,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_add_transpose_kernel(
    x_ptr,           # in_0: [128, 1] - bias tensor
    y_ptr,           # in_1: [1, 128, 19] - input tensor
    out_ptr,         # output: [1, 19, 128]
    n_seqlen,        # 19
    n_hidden,        # 128
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the output tensor
    # Output shape: [1, 19, 128]
    seq_idx = tl.program_id(0)  # 0..18
    hidden_idx = tl.program_id(1)  # 0..127
    
    # Calculate the linear index in the output tensor
    # Output shape: [1, 19, 128]
    output_offset = seq_idx * n_hidden + hidden_idx
    
    # Load from y_ptr (in_1): [1, 128, 19] -> reshape to [128, 19]
    # We need y[hidden_idx, seq_idx]
    y_offset = hidden_idx * n_seqlen + seq_idx
    y_val = tl.load(y_ptr + y_offset, mask=(y_offset < n_hidden * n_seqlen), other=0.0)
    
    # Load from x_ptr (in_0): [128, 1] -> we need x[hidden_idx, 0]
    x_offset = hidden_idx
    x_val = tl.load(x_ptr + x_offset, mask=(x_offset < n_hidden), other=0.0)
    
    # Perform addition
    out_val = x_val + y_val
    
    # Store to out_ptr: [1, 19, 128]
    tl.store(out_ptr + output_offset, out_val, mask=(output_offset < n_hidden * n_seqlen))

@torch.fx.wrap
def fused_add_transpose(x, y):
    # Input shapes: x=[128, 1], y=[1, 128, 19]
    n_hidden, n_seqlen = x.shape[0], y.shape[2]
    
    # Output shape: [1, 19, 128]
    output_shape = (1, n_seqlen, n_hidden)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Flatten the output tensor for the kernel
    out_ptr = out.reshape(-1)
    
    # Flatten the input y tensor: [1, 128, 19] -> [128, 19]
    y_flat = y.reshape(n_hidden, n_seqlen)
    
    BLOCK_SIZE = 1024
    # Calculate grid dimensions: [seqlen, hidden] = [19, 128]
    grid = lambda meta: (n_seqlen, n_hidden)
    
    fused_add_transpose_kernel[grid](
        x_ptr=x.reshape(-1),
        y_ptr=y_flat,
        out_ptr=out_ptr,
        n_seqlen=n_seqlen,
        n_hidden=n_hidden,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_add_transpose