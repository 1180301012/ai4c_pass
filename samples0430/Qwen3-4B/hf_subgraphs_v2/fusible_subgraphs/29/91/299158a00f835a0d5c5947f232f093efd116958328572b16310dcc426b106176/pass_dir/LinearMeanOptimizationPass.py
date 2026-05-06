import torch
import triton
import triton.language as tl


def pattern(in0, in1, in2, in3):
    linear_out = torch.nn.functional.linear(in2, in1, in0)
    mean_out = in3.mean(-2)
    return linear_out, mean_out

def replacement_args(in0, in1, in2, in3):
    return in0, in1, in2, in3

@triton.jit
def optimized_kernel(
    in2_ptr,
    in1_ptr,
    in0_ptr,
    in3_ptr,
    out_linear_ptr,
    out_mean_ptr,
    in2_shape,
    in1_shape,
    in0_shape,
    in3_shape,
    BLOCK_SIZE: tl.constexpr = 128,
):
    # Unroll the linear layer computation
    batch_id = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE)
    
    # Linear layer: compute B x K matrix
    out_linear = tl.zeros((BLOCK_SIZE, in1_shape[1]), dtype=tl.float32)
    
    # For each output feature
    for k in range(in1_shape[1]):
        # Compute contribution from each input feature
        for f in range(in1_shape[0]):
            val = tl.load(in2_ptr + (batch_id * in2_shape[1] + offset) * in1_shape[1] + f)
            val = val * tl.load(in1_ptr + f * in1_shape[1] + k)
            tl.store(out_linear + offset + k, val, mask=offset < BLOCK_SIZE)
        
        # Add bias
        bias_val = tl.load(in0_ptr + k)
        tl.store(out_linear + offset + k, out_linear + bias_val, mask=offset < BLOCK_SIZE)
    
    # Mean operation: compute over sequence dimension
    out_mean = tl.zeros((BLOCK_SIZE, in3_shape[2]), dtype=tl.float32)
    for f in range(in3_shape[2]):
        total = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for s in range(in3_shape[1]):
            val = tl.load(in3_ptr + (batch_id * in3_shape[1] * in3_shape[2] + s * in3_shape[2] + f))
            total += val
        mean_val = total / in3_shape[1]
        tl.store(out_mean + offset + f, mean_val, mask=offset < BLOCK_SIZE)
    
    # Store results
    tl.store(out_linear_ptr + (batch_id * in1_shape[1] + offset), out_linear, mask=offset < BLOCK_SIZE)
    tl.store(out_mean_ptr + (batch_id * in3_shape[2] + offset), out_mean, mask=offset < BLOCK_SIZE)

@torch.fx.wrap
def kernel_wrapper(in0, in1, in2, in3):
    B, F = in2.shape
    K = in0.shape[0]
    S = in3.shape[1]
    
    out_linear = torch.empty((B, K), dtype=in0.dtype, device=in0.device)
    out_mean = torch.empty((B, F), dtype=in3.dtype, device=in3.device)
    
    optimized_kernel[(B,)](
        in2_ptr=in2,
        in1_ptr=in1,
        in0_ptr=in0,
        in3_ptr=in3,
        out_linear_ptr=out_linear,
        out_mean_ptr=out_mean,
        in2_shape=(B, F),
        in1_shape=(F, K),
        in0_shape=(K,),
        in3_shape=(B, S, F),
        BLOCK_SIZE=128,
    )
    
    return out_linear, out_mean

def replacement_func():
    return kernel_wrapper