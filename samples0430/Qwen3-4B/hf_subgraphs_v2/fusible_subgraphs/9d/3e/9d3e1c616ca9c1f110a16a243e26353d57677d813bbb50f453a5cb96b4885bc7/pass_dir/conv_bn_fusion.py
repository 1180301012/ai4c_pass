import torch
import triton
import triton.language as tl

def pattern(x, y, z, w, v, u, t):
    conv = torch.conv2d(t, v, None, (1, 1), (0, 0), (1, 1), 1)
    bn = torch.nn.functional.batch_norm(conv, x, y, w, z, False, 0.1, 1e-05)
    return bn + u

def replacement_args(x, y, z, w, v, u, t):
    return x, y, z, w, v, u, t

@triton.jit
def conv_bn_kernel(
    x_ptr,  # running_mean (C_out)
    y_ptr,  # running_var (C_out)
    z_ptr,  # bias (C_out)
    w_ptr,  # weight (C_out)
    v_ptr,  # conv weights (C_out, C_in)
    u_ptr,  # add tensor (C_out, H, W)
    t_ptr,  # input (C_in, H, W)
    out_ptr,  # output (C_out, H, W)
    n_channels: tl.constexpr,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * block_size
    mask = tl.arange(0, block_size) < n_channels

    # Load running statistics
    running_means = tl.load(x_ptr + offset, mask=mask, other=0.0)
    running_vars = tl.load(y_ptr + offset, mask=mask, other=0.0)
    bias = tl.load(z_ptr + offset, mask=mask, other=0.0)
    weight_bn = tl.load(w_ptr + offset, mask=mask, other=0.0)

    # Load input and conv weights
    inputs = tl.load(t_ptr + offset, mask=mask, other=0.0)
    conv_weights = tl.load(v_ptr + offset, mask=mask, other=0.0)

    # Conv2D (1x1) → element-wise multiplication
    conv = inputs * conv_weights

    # BatchNorm (inference mode)
    eps = 1e-05
    var_sqrt = tl.sqrt(running_vars + eps)
    temp = (conv - running_means) / var_sqrt
    temp = temp * weight_bn + bias

    # Add the add tensor
    add_vals = tl.load(u_ptr + offset, mask=mask, other=0.0)
    out = temp + add_vals

    # Store result
    tl.store(out_ptr + offset, out, mask=mask)

def kernel_wrapper(x, y, z, w, v, u, t):
    B, C_in, H, W = t.shape
    C_out = v.shape[0]
    out = torch.empty_like(t)
    
    # Configuration for the kernel
    BLOCK_SIZE = 128
    num_blocks = (C_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    conv_bn_kernel[ (num_blocks,) ](
        x_ptr=x,
        y_ptr=y,
        z_ptr=z,
        w_ptr=w,
        v_ptr=v,
        u_ptr=u,
        t_ptr=t,
        out_ptr=out,
        n_channels=C_out,
        block_size=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return kernel_wrapper