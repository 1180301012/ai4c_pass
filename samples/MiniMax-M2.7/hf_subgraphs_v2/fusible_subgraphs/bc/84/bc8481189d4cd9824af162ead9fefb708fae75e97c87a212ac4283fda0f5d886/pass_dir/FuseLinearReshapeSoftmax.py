import torch
import triton
import triton.language as tl

# Pattern matching function - matches linear + reshape + softmax
def pattern(in_0, in_1, in_2):
    """
    Match the pattern: linear -> reshape -> softmax
    """
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.reshape(linear, [-1, 9, 1])
    tmp_4 = torch.softmax(tmp_3, dim = 1)
    return tmp_4

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_linear_reshape_softmax_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    R: tl.constexpr,
    S: tl.constexpr,
    Cin: tl.constexpr,
    Cout: tl.constexpr
):
    """
    Fused kernel for linear + reshape + softmax computation.
    Each program computes one row of the softmax output.
    """
    # Get program index
    n = tl.program_id(0)
    
    # Compute linear outputs for all R elements in this row
    # flat_idx = n * R + r maps to linear[b=0, s=flat_idx//Cout, c=flat_idx%Cout]
    
    for r in range(R):
        flat_idx = n * R + r
        s_idx = flat_idx // Cout  # Sequence position
        c_idx = flat_idx % Cout   # Output channel
        
        # Initialize accumulator with bias
        acc = tl.load(bias_ptr + c_idx).to(tl.float32)
        
        # Inner loop: compute dot product for this (s_idx, c_idx) position
        for cin in range(Cin):
            input_val = tl.load(input_ptr + s_idx * Cin + cin).to(tl.float32)
            weight_val = tl.load(weight_ptr + c_idx * Cin + cin).to(tl.float32)
            acc = acc + input_val * weight_val
        
        # Store linear result
        tl.store(output_ptr + flat_idx, acc)
    
    # Softmax over R elements for this row
    # Load values and compute softmax directly
    base_offset = n * R
    
    # Load the 9 values and convert to fp32 for softmax computation
    x0 = tl.load(output_ptr + base_offset + 0).to(tl.float32)
    x1 = tl.load(output_ptr + base_offset + 1).to(tl.float32)
    x2 = tl.load(output_ptr + base_offset + 2).to(tl.float32)
    x3 = tl.load(output_ptr + base_offset + 3).to(tl.float32)
    x4 = tl.load(output_ptr + base_offset + 4).to(tl.float32)
    x5 = tl.load(output_ptr + base_offset + 5).to(tl.float32)
    x6 = tl.load(output_ptr + base_offset + 6).to(tl.float32)
    x7 = tl.load(output_ptr + base_offset + 7).to(tl.float32)
    x8 = tl.load(output_ptr + base_offset + 8).to(tl.float32)
    
    # Find max for numerical stability
    x_max = x0
    x_max = tl.where(x1 > x_max, x1, x_max)
    x_max = tl.where(x2 > x_max, x2, x_max)
    x_max = tl.where(x3 > x_max, x3, x_max)
    x_max = tl.where(x4 > x_max, x4, x_max)
    x_max = tl.where(x5 > x_max, x5, x_max)
    x_max = tl.where(x6 > x_max, x6, x_max)
    x_max = tl.where(x7 > x_max, x7, x_max)
    x_max = tl.where(x8 > x_max, x8, x_max)
    
    # Compute exp(x - max) for each value
    exp0 = tl.exp(x0 - x_max)
    exp1 = tl.exp(x1 - x_max)
    exp2 = tl.exp(x2 - x_max)
    exp3 = tl.exp(x3 - x_max)
    exp4 = tl.exp(x4 - x_max)
    exp5 = tl.exp(x5 - x_max)
    exp6 = tl.exp(x6 - x_max)
    exp7 = tl.exp(x7 - x_max)
    exp8 = tl.exp(x8 - x_max)
    
    # Sum of exponentials
    sum_exp = exp0 + exp1 + exp2 + exp3 + exp4 + exp5 + exp6 + exp7 + exp8
    
    # Compute softmax
    out0 = exp0 / sum_exp
    out1 = exp1 / sum_exp
    out2 = exp2 / sum_exp
    out3 = exp3 / sum_exp
    out4 = exp4 / sum_exp
    out5 = exp5 / sum_exp
    out6 = exp6 / sum_exp
    out7 = exp7 / sum_exp
    out8 = exp8 / sum_exp
    
    # Store final softmax result
    tl.store(output_ptr + base_offset + 0, out0)
    tl.store(output_ptr + base_offset + 1, out1)
    tl.store(output_ptr + base_offset + 2, out2)
    tl.store(output_ptr + base_offset + 3, out3)
    tl.store(output_ptr + base_offset + 4, out4)
    tl.store(output_ptr + base_offset + 5, out5)
    tl.store(output_ptr + base_offset + 6, out6)
    tl.store(output_ptr + base_offset + 7, out7)
    tl.store(output_ptr + base_offset + 8, out8)


@torch.fx.wrap
def fused_operations_wrapper(in_0, in_1, in_2):
    """
    Wrapper function for fused linear + reshape + softmax operation.
    This function is called when the pattern matches and replaces the
    original computation.
    """
    # Extract dimensions
    B = in_2.shape[0]   # Batch size (1)
    S = in_2.shape[1]   # Sequence length (19)  
    Cin = in_2.shape[2] # Input features (128)
    Cout = in_1.shape[0] # Output features (18)
    
    # Compute derived dimensions
    # After linear: [B, S, Cout] = [1, 19, 18]
    # Reshape to [-1, 9, 1] => [38, 9, 1]
    R = 9
    num_rows = B * S * Cout // R  # 342 / 9 = 38
    
    # Allocate output tensor with correct shape and dtype
    output = torch.empty(
        (num_rows, R, 1),
        dtype=in_2.dtype,
        device=in_2.device
    )
    
    # Launch Triton kernel
    # Grid: one block per softmax row
    grid = (num_rows,)
    
    fused_linear_reshape_softmax_kernel[grid](
        in_2, in_1, in_0,
        output,
        R, S, Cin, Cout
    )
    
    return output


def replacement_func():
    return fused_operations_wrapper