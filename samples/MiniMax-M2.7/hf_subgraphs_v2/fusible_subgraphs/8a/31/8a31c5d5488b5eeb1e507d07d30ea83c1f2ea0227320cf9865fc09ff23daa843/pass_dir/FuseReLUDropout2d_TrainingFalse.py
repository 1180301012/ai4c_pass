import torch
import triton
import triton.language as tl

# Pattern matching function
# Matches: relu(inplace=True) -> dropout2d(training=False, inplace=False)
def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.dropout2d(tmp_0, 0.1, False, False)
    return tmp_1, tmp_0

# Argument extraction function
def replacement_args(in_0):
    return (in_0, 0.1, False, False)

# Optimized fused kernel for ReLU + Dropout2d (training=False)
@triton.jit
def fused_relu_dropout2d_kernel(
    inp_ptr,
    out_ptr,
    p,
    training,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU (this is the only operation needed when training=False)
    x = tl.where(x > 0, x, 0)
    
    # When training=False, dropout is a no-op - just store the result
    # The mask is all 1s, so output = input / (1-p) = input (since input >= 0 after ReLU)
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def fused_relu_dropout2d(in_0, p, training, inplace):
    """
    Fused ReLU + Dropout2d kernel.
    
    When training=False, dropout2d is a no-op that returns input unchanged.
    This kernel fuses ReLU with the identity dropout, eliminating kernel call
    overhead and unnecessary memory allocation.
    
    Since relu output is always non-negative, and we need to return both
    dropout_output and relu_output (which are identical when training=False),
    we simply apply ReLU once and return the same tensor for both outputs.
    """
    n_elements = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor (inplace semantics not needed since we return same value)
    out = torch.empty_like(in_0)
    
    fused_relu_dropout2d_kernel[(num_programs,)](
        inp_ptr=in_0,
        out_ptr=out,
        p=p,
        training=training,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # When training=False, dropout2d returns input unchanged.
    # Both outputs are identical: relu result.
    # We need to return (dropout_output, relu_output) = (out, out)
    # Since out is the relu result, and dropout doesn't change it,
    # we simply return (out, out) - two references to the same tensor.
    return out, out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_relu_dropout2d