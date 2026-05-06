import torch
import triton
import triton.language as tl

def pattern(x, weight, bias, running_mean, running_var, weight_bn, bias_bn,
             input1, input2, output_channels):
    conv_out = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), output_channels)
    add1 = input1 + conv_out
    add2 = add1 + input2
    batch_norm = torch.nn.functional.batch_norm(add2, running_mean, running_var, weight_bn, bias_bn, 
                                             training=False, momentum=0.1, eps=1e-05)
    mean_pooled = batch_norm.mean((2, 3), keepdim=True)
    return batch_norm, mean_pooled

def replacement_args(x, weight, bias, running_mean, running_var, weight_bn, bias_bn,
                     input1, input2, output_channels):
    return (x, weight, bias, running_mean, running_var, weight_bn, bias_bn,
            input1, input2, output_channels)

@triton.jit
def optimized_kernel(x_ptr, weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr, 
                     weight_bn_ptr, bias_bn_ptr, input1_ptr, input2_ptr,
                     output_channels, N, H, W, BLOCK_SIZE, GROUPS):
    # Output tensor allocation
    output = tl.zeros((BLOCK_SIZE, N, H, W), dtype=tl.float32)
    # Initialize the program
    for i in range(BLOCK_SIZE):
        # Load inputs
        x = tl.load(x_ptr + i, mask=i < N)
        weight = tl.load(weight_ptr + i, mask=i < N)
        bias = tl.load(bias_ptr + i, mask=i < N)
        running_mean = tl.load(running_mean_ptr + i, mask=i < N)
        running_var = tl.load(running_var_ptr + i, mask=i < N)
        weight_bn = tl.load(weight_bn_ptr + i, mask=i < N)
        bias_bn = tl.load(bias_bn_ptr + i, mask=i < N)
        input1 = tl.load(input1_ptr + i, mask=i < N)
        input2 = tl.load(input2_ptr + i, mask=i < N)
        
        # Perform convolution
        conv_out = x * weight + bias
        
        # Additions
        add1 = input1 + conv_out
        add2 = add1 + input2
        
        # Batch normalization
        batch_norm = (add2 * (running_mean * weight_bn)) + bias_bn
        
        # Mean pooling over spatial dimensions
        mask = tl.arange(0, H) < tl.arange(0, H)[:, None]
        mean = tl.sum(batch_norm, axis=(2, 3))
        mean = mean / (H * W)
        
        # Store result
        tl.store(output + i, mean)

@torch.fx.wrap
def kernel_wrapper(x, weight, bias, running_mean, running_var, weight_bn, bias_bn,
                    input1, input2, output_channels):
    N, H, W = x.shape[0], x.shape[1], x.shape[2]
    BLOCK_SIZE = 128
    output = torch.empty_like(x)
    # Call with grid
    return optimized_kernel[ (1, 1) ](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_bn_ptr=weight_bn,
        bias_bn_ptr=bias_bn,
        input1_ptr=input1,
        input2_ptr=input2,
        output_channels=output_channels,
        N=N,
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUPS=1
    )

def replacement_func():
    return kernel_wrapper