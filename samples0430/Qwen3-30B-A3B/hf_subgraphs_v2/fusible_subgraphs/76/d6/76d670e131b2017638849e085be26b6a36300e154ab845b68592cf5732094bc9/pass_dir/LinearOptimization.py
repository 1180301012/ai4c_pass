import torch
import triton
import triton.language as tl

def pattern(in_6, in_5, in_4):
    return torch.nn.functional.linear(in_6, in_5, in_4)

def replacement_args(in_6, in_5, in_4):
    return (in_6, in_5, in_4)

@triton.jit
def linear_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    batch,
    in_features,
    out_features,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_IN: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_out = tl.program_id(1)
    
    block_start = pid_out * BLOCK_OUT
    block_end = min(block_start + BLOCK_OUT, out_features)
    out_range = block_end - block_start
    
    for start_in in range(0, in_features, BLOCK_IN):
        w_row_offsets = start_in + tl.arange(0, BLOCK_IN)
        w_col_offsets = pid_out * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        w_offsets = w_row_offsets[:, None] * in_features + w_col_offsets[None, :]
        w = tl.load(w_ptr + w_offsets, 
                    mask=(w_row_offsets[:, None] < out_features) & (w_col_offsets[None, :] < in_features))
        
        x_row_offsets = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
        x_col_offsets = start_in + tl.arange(0, BLOCK_IN)
        x_offsets = x_row_offsets[:, None] * in_features + x_col_offsets[None, :]
        x = tl.load(x_ptr + x_offsets, 
                    mask=(x_row_offsets[:, None] < batch) & (x_col_offsets[None, :] < in_features))
        
        acc = tl.zeros((BLOCK_BATCH, BLOCK_OUT), dtype=tl.float32)
        acc += tl.dot(x, w, allow_tf32=True)
        
        b = tl.load(b_ptr + block_start + tl.arange(0, BLOCK_OUT), 
                    mask=(block_start + tl.arange(0, BLOCK_OUT) < out_features))
        
        out_row_offsets = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
        out_col_offsets = block_start + tl.arange(0, BLOCK_OUT)
        out_offsets = out_row_offsets[:, None] * out_features + out_col_offsets[None, :]
        out = acc + b[None, :]
        tl.store(out_ptr + out_offsets, out, 
                mask=(out_row_offsets[:, None] < batch) & (out_col_offsets[None, :] < out_features))

@torch.fx.wrap
def optimized_linear(x, w, b):
    batch, in_features = x.shape
    out_features = w.shape[0]
    
    BLOCK_BATCH = 64
    BLOCK_OUT = 32
    BLOCK_IN = 64
    
    grid = (triton.cdiv(batch, BLOCK_BATCH), triton.cdiv(out_features, BLOCK_OUT))
    
    out = torch.empty((batch, out_features), device=x.device, dtype=torch.float32)
    linear_kernel[grid](
        x_ptr=x,
        w_ptr=w,
        b_ptr=b,
        out_ptr=out,
        batch=batch,
        in_features=in_features,
        out_features=out_features,
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_OUT=BLOCK_OUT,
        BLOCK_IN=BLOCK_IN
    )
    
    return out.to(x.dtype)

def replacement_func():
    return optimized_linear