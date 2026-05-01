import torch
import triton
import triton.language as tl

def pattern(in_3, in_1, in_0):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    return linear

def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

@triton.jit
def linear_kernel(input_ptr, weight_ptr, bias_ptr, output_ptr, in_features, out_features, BLOCK_SIZE_OUT: tl.constexpr, TILE_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE_OUT
    offsets_out = block_start + tl.arange(0, BLOCK_SIZE_OUT)
    mask_out = offsets_out < out_features

    acc = tl.zeros((BLOCK_SIZE_OUT,), dtype=tl.float32)
    for i in range(0, in_features, TILE_SIZE):
        input_tile = tl.load(input_ptr + i + tl.arange(0, TILE_SIZE), 
                          mask=(i + tl.arange(0, TILE_SIZE) < in_features), 
                          other=0.0)
        weight_tile = tl.load(weight_ptr + (block_start * in_features + i), 
                            shape=(BLOCK_SIZE_OUT, TILE_SIZE), 
                            stride=(in_features, 1),
                            mask=(mask_out[:, None], (i + tl.arange(0, TILE_SIZE) < in_features)[None, :]),
                            other=0.0)
        acc += tl.dot(weight_tile, input_tile)

    bias = tl.load(bias_ptr + offsets_out, mask=mask_out, other=0.0)
    acc += bias
    tl.store(output_ptr + offsets_out, acc, mask=mask_out)

@torch.fx.wrap
def linear_triton_kernel(input, weight, bias):
    input_flat = input.view(-1)
    in_features = input_flat.numel()
    out_features = weight.shape[0]
    
    output_flat = torch.empty(out_features, dtype=input.dtype)
    BLOCK_SIZE_OUT = 64
    TILE_SIZE = 64
    num_blocks = (out_features + BLOCK_SIZE_OUT - 1) // BLOCK_SIZE_OUT
    
    linear_kernel[(num_blocks,)](
        input_ptr=input_flat,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output_flat,
        in_features=in_features,
        out_features=out_features,
        BLOCK_SIZE_OUT=BLOCK_SIZE_OUT,
        TILE_SIZE=TILE_SIZE
    )
    
    return output_flat.view(1, 1, -1)

def replacement_func():
    return linear_triton_kernel