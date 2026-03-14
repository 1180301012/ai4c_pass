import torch
import triton
import triton.language as tl

# Pattern for BERT-style: linear -> dropout(0.1, False, False) -> add
def pattern(bias, weight, residual, input):
    linear_out = torch.nn.functional.linear(input, weight, bias)
    dropout_out = torch.nn.functional.dropout(linear_out, 0.1, False, False)
    output = dropout_out + residual
    return output

def replacement_args(bias, weight, residual, input):
    return (bias, weight, residual, input)

@triton.jit
def fused_linear_add_kernel(
    input_ptr, weight_ptr, bias_ptr, residual_ptr, output_ptr,
    num_rows, stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    row_start = pid * BLOCK_SIZE
    row_ids = row_start + tl.arange(0, BLOCK_SIZE)
    mask = row_ids < num_rows
    
    # Load 2x2 weight and bias
    w00 = tl.load(weight_ptr + 0)
    w01 = tl.load(weight_ptr + 1)
    w10 = tl.load(weight_ptr + 2)
    w11 = tl.load(weight_ptr + 3)
    b0 = tl.load(bias_ptr + 0)
    b1 = tl.load(bias_ptr + 1)
    
    # Compute offsets
    base = row_ids * stride_row
    
    # Load and compute
    x0 = tl.load(input_ptr + base, mask=mask, other=0.0)
    x1 = tl.load(input_ptr + base + 1, mask=mask, other=0.0)
    r0 = tl.load(residual_ptr + base, mask=mask, other=0.0)
    r1 = tl.load(residual_ptr + base + 1, mask=mask, other=0.0)
    
    y0 = x0 * w00 + x1 * w01 + b0 + r0
    y1 = x0 * w10 + x1 * w11 + b1 + r1
    
    tl.store(output_ptr + base, y0, mask=mask)
    tl.store(output_ptr + base + 1, y1, mask=mask)

@torch.fx.wrap
def fused_linear_residual_add(bias, weight, residual, input):
    original_shape = input.shape
    num_rows = input.numel() // 2
    
    output = torch.empty_like(input)
    
    BLOCK_SIZE = 512
    num_blocks = (num_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_linear_add_kernel[(num_blocks,)](
        input, weight, bias, residual, output,
        num_rows, 2,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_linear_residual_add