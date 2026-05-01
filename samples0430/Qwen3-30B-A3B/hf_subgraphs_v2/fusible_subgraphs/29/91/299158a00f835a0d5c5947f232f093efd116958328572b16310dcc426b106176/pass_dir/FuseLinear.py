import torch
import triton
import triton.language as tl

@triton.jit
def linear_kernel(
    x_ptr, weight_ptr, bias_ptr,
    out_ptr,
    batch_size, input_size, output_size,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_INPUT: tl.constexpr,
    BLOCK_OUTPUT: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_output = tl.program_id(1)
    
    batch_start = pid_batch * BLOCK_BATCH
    output_start = pid_output * BLOCK_OUTPUT
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_BATCH, BLOCK_OUTPUT), dtype=tl.float32)
    
    # Iterate over input dimension
    for k in range(0, input_size, BLOCK_INPUT):
        # Load a block of input x
        x = tl.load(
            x_ptr + batch_start * input_size + k,
            shape=(BLOCK_BATCH, min(BLOCK_INPUT, input_size - k)),
            mask=(batch_start + tl.arange(0, BLOCK_BATCH) < batch_size) & 
                 (k + tl.arange(0, BLOCK_INPUT) < input_size),
            other=0.0
        )
        
        # Load a block of weight (for the weight's transpose)
        weight = tl.load(
            weight_ptr + k * output_size + output_start,
            shape=(min(BLOCK_INPUT, input_size - k), BLOCK_OUTPUT),
            mask=(k + tl.arange(0, BLOCK_INPUT) < input_size) & 
                 (output_start + tl.arange(0, BLOCK_OUTPUT) < output_size),
            other=0.0
        )
        
        # Compute the dot product for this block
        accumulator += tl.dot(x, weight)
    
    # Add bias
    bias = tl.load(
        bias_ptr + output_start,
        shape=(BLOCK_OUTPUT,),
        mask=(output_start + tl.arange(0, BLOCK_OUTPUT) < output_size),
        other=0.0
    )
    accumulator += bias
    
    # Store the result
    tl.store(
        out_ptr + batch_start * output_size + output_start,
        accumulator,
        mask=(batch_start + tl.arange(0, BLOCK_BATCH) < batch_size) & 
             (output_start + tl.arange(0, BLOCK_OUTPUT) < output_size)
    )

@torch.fx.wrap
def triton_linear(x, weight, bias):
    batch_size, input_size = x.shape
    output_size = weight.shape[0]
    
    out = torch.empty((batch_size, output_size), dtype=x.dtype, device=x.device)
    
    BLOCK_BATCH = 128
    BLOCK_INPUT = 64
    BLOCK_OUTPUT = 2
    
    num_batches = (batch_size + BLOCK_BATCH - 1) // BLOCK_BATCH
    num_outputs = (output_size + BLOCK_OUTPUT - 1) // BLOCK_OUTPUT
    
    linear_kernel[(num_batches, num_outputs)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch_size=batch_size,
        input_size=input_size,
        output_size=output_size,
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_INPUT=BLOCK_INPUT,
        BLOCK_OUTPUT=BLOCK_OUTPUT
    )
    
    return out

def pattern(x, weight, bias):
    return torch.nn.functional.linear(x, weight, bias)

def replacement_args(x, weight, bias):
    return (x, weight, bias)

def replacement_func():
    return triton_linear