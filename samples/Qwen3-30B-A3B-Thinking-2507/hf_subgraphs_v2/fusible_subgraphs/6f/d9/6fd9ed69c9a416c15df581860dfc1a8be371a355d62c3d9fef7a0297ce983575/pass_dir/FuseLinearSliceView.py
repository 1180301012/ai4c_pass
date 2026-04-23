import torch
import triton
import triton.language as tl

def pattern(in_5, in_0, in_1):
    linear = torch.nn.functional.linear(in_5, in_1, in_0)
    slice1 = linear[:, :256]
    view1 = slice1.view(-1, 256)
    slice2 = linear[:, -256:]
    view2 = slice2.view(-1, 256)
    return view1, view2

def replacement_args(in_5, in_0, in_1):
    weight_half1 = in_1[:256, :]
    weight_half2 = in_1[256:, :]
    bias_half1 = in_0[:256]
    bias_half2 = in_0[256:]
    return (in_5, weight_half1, weight_half2, bias_half1, bias_half2)

@triton.jit
def fused_linear_kernel(
    x_ptr,
    w1_ptr,
    w2_ptr,
    b1_ptr,
    b2_ptr,
    out1_ptr,
    out2_ptr,
    batch_size,
    input_features,
    output_features,
    BLOCK_BATCH: tl.constexpr = 32,
    BLOCK_OUTPUT: tl.constexpr = 32,
    BLOCK_INPUT: tl.constexpr = 32
):
    batch_idx = tl.program_id(0) * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    batch_mask = batch_idx < batch_size
    
    output_idx = tl.program_id(1) * BLOCK_OUTPUT + tl.arange(0, BLOCK_OUTPUT)
    output_mask = output_idx < output_features

    for k in range(0, input_features, BLOCK_INPUT):
        x = tl.load(
            x_ptr + (batch_idx[:, None] * input_features + k),
            mask=batch_mask[:, None] & (k + tl.arange(0, BLOCK_INPUT) < input_features),
            other=0.0
        )
        
        w1 = tl.load(
            w1_ptr + (output_idx[:, None] * input_features + k),
            mask=output_mask[:, None] & (k + tl.arange(0, BLOCK_INPUT) < input_features),
            other=0.0
        )
        
        w2 = tl.load(
            w2_ptr + (output_idx[:, None] * input_features + k),
            mask=output_mask[:, None] & (k + tl.arange(0, BLOCK_INPUT) < input_features),
            other=0.0
        )
        
        out1 = tl.dot(x, w1)
        out2 = tl.dot(x, w2)
        
        bias1 = tl.load(b1_ptr + output_idx, mask=output_mask)
        bias2 = tl.load(b2_ptr + output_idx, mask=output_mask)
        
        out1 = out1 + bias1
        out2 = out2 + bias2
        
        tl.store(
            out1_ptr + (batch_idx[:, None] * output_features + output_idx),
            out1,
            mask=batch_mask[:, None] & output_mask[None, :]
        )
        tl.store(
            out2_ptr + (batch_idx[:, None] * output_features + output_idx),
            out2,
            mask=batch_mask[:, None] & output_mask[None, :]
        )

@torch.fx.wrap
def fused_linear(in_5, weight_half1, weight_half2, bias_half1, bias_half2):
    batch_size, input_features = in_5.shape
    output_features = weight_half1.shape[0]
    
    out1 = torch.empty(batch_size, output_features, dtype=in_5.dtype, device=in_5.device)
    out2 = torch.empty(batch_size, output_features, dtype=in_5.dtype, device=in_5.device)
    
    BLOCK_BATCH = 32
    BLOCK_OUTPUT = 32
    BLOCK_INPUT = 32
    
    grid_batch = (batch_size + BLOCK_BATCH - 1) // BLOCK_BATCH
    grid_output = (output_features + BLOCK_OUTPUT - 1) // BLOCK_OUTPUT
    
    fused_linear_kernel[(grid_batch, grid_output)](
        x_ptr=in_5,
        w1_ptr=weight_half1,
        w2_ptr=weight_half2,
        b1_ptr=bias_half1,
        b2_ptr=bias_half2,
        out1_ptr=out1,
        out2_ptr=out2,
        batch_size=batch_size,
        input_features=input_features,
        output_features=output_features,
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_OUTPUT=BLOCK_OUTPUT,
        BLOCK_INPUT=BLOCK_INPUT
    )
    
    return out1, out2

def replacement_func():
    return fused_linear