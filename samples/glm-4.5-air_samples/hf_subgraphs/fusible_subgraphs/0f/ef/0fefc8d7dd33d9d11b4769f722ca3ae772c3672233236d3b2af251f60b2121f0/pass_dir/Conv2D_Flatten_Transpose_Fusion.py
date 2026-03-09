import torch
import triton
import triton.language as tl

def pattern(input, weight, bias, kernel_size, padding, stride, dilation, groups=1):
    # For pattern matching, we need to reconstruct the exact conv2d call as in the graph
    # The graph calls conv2d with the specific argument order: input, weight, bias, kernel_size, padding, stride, dilation
    conv_out = torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)
    flatten_out = conv_out.flatten(2)
    transpose_out = flatten_out.transpose(1, 2)
    return transpose_out

def replacement_args(input, weight, bias, kernel_size, padding, stride, dilation, groups=1):
    return (input, weight, bias, kernel_size, padding, stride, dilation, groups)

@triton.jit
def conv_patch_embedding_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    N, C_in, C_out, H_in, W_in,
    KH, KW, SH, SW, PH, PW,
    OUTPUT_SEQ_LEN,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    
    # Pointers
    input_ptr = input_ptr + pid_n * C_in * H_in * W_in
    weight_ptr = weight_ptr + pid_c_out * C_in * KH * KW
    output_ptr = output_ptr + pid_n * OUTPUT_SEQ_LEN * C_out + pid_c_out * OUTPUT_SEQ_LEN
    
    # Load bias if needed
    bias = 0.0
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + pid_c_out)
    
    # Convolution computation
    for h_idx in range((H_in - KH + 2 * PH) // SH):
        for w_idx in range((W_in - KW + 2 * PW) // SW):
            # Output position
            seq_idx = h_idx * ((W_in - KW + 2 * PW) // SW) + w_idx
            
            # Initialize accumulator
            acc = bias
            
            # Compute convolution
            for c_in in range(C_in):
                for kh in range(KH):
                    for kw in range(KW):
                        h_in = h_idx * SH + kh - PH
                        w_in = w_idx * SW + kw - PW
                        
                        if 0 <= h_in < H_in and 0 <= w_in < W_in:
                            input_val = tl.load(input_ptr + c_in * H_in * W_in + h_in * W_in + w_in)
                            weight_val = tl.load(weight_ptr + c_in * KH * KW + kh * KW + kw)
                            acc += input_val * weight_val
            
            # Store output
            if seq_idx < OUTPUT_SEQ_LEN:
                tl.store(output_ptr + seq_idx, acc)

@torch.fx.wrap
def fused_conv_patch_embedding(input, weight, bias, kernel_size, padding, stride, dilation, groups=1):
    N, C_in, H_in, W_in = input.shape
    C_out, _, KH, KW = weight.shape
    
    # Calculate output dimensions
    H_out = (H_in - KH + 2 * padding[0]) // stride[0] + 1
    W_out = (W_in - KW + 2 * padding[1]) // stride[1] + 1
    OUTPUT_SEQ_LEN = H_out * W_out
    
    # Create output tensor
    output = torch.empty(N, OUTPUT_SEQ_LEN, C_out, dtype=input.dtype, device=input.device)
    
    # Determine block sizes based on tensor dimensions
    BLOCK_SIZE_N = min(64, N)
    BLOCK_SIZE_C = min(64, C_out)
    BLOCK_SIZE_HW = min(256, OUTPUT_SEQ_LEN)
    
    # Grid setup
    grid = (
        (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
        (C_out + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C,
    )
    
    # Launch kernel
    conv_patch_embedding_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        N=N, C_in=C_in, C_out=C_out, H_in=H_in, W_in=W_in,
        KH=kernel_size[0], KW=kernel_size[1],
        SH=stride[0], SW=stride[1],
        PH=padding[0], PW=padding[1],
        OUTPUT_SEQ_LEN=OUTPUT_SEQ_LEN,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
    )
    
    return output

def replacement_func():
    return fused_conv_patch_embedding