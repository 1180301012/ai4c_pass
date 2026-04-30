import torch
import triton
import triton.language as tl

@triton.jit
def fused_add_mean_kernel_2in(
    x0_ptr, x1_ptr,
    out_ptr, mean_ptr,
    n_elements,
    hw_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a channel
    pid = tl.program_id(0)
    
    # offsets for H*W elements in this channel
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hw_size
    
    # Load both inputs (N, C, H, W) - load at channel pid
    x0 = tl.load(x0_ptr + pid * hw_size + offs, mask=mask, other=0.0)
    x1 = tl.load(x1_ptr + pid * hw_size + offs, mask=mask, other=0.0)
    
    # Compute sum
    s = x0 + x1
    
    # Compute mean
    mean_val = tl.sum(s, axis=0) / hw_size
    
    # Store output (add result) and mean result
    tl.store(out_ptr + pid * hw_size + offs, s, mask=mask)
    tl.store(mean_ptr + pid, mean_val)

@torch.fx.wrap
def fused_add_mean_2in(x0, x1):
    """
    Fused kernel: (x0 + x1).mean(dim=[2,3], keepdim=True)
    Optimizes by:
    1. Fusing add and mean into single kernel
    2. Avoiding intermediate tensor allocation
    """
    n, c, h, w = x0.shape
    
    # Allocate output tensors
    out = torch.empty_like(x0)
    mean_out = torch.empty((n, c, 1, 1), dtype=x0.dtype, device=x0.device)
    
    # Launch kernel - one program per channel
    grid = (n * c,)
    BLOCK_SIZE = 1024
    hw_size = h * w
    
    fused_add_mean_kernel_2in[grid](
        x0, x1,
        out, mean_out,
        n * c * h * w,
        hw_size,
        BLOCK_SIZE,
    )
    
    return out, mean_out


@triton.jit
def fused_add_mean_kernel_3in(
    x0_ptr, x1_ptr, x2_ptr,
    out_ptr, mean_ptr,
    n_elements,
    hw_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a channel
    pid = tl.program_id(0)
    
    # offsets for H*W elements in this channel
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hw_size
    
    # Load all three inputs (N, C, H, W) - load at channel pid
    x0 = tl.load(x0_ptr + pid * hw_size + offs, mask=mask, other=0.0)
    x1 = tl.load(x1_ptr + pid * hw_size + offs, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + pid * hw_size + offs, mask=mask, other=0.0)
    
    # Compute sum
    s = x0 + x1 + x2
    
    # Compute mean
    mean_val = tl.sum(s, axis=0) / hw_size
    
    # Store output (add result) and mean result
    tl.store(out_ptr + pid * hw_size + offs, s, mask=mask)
    tl.store(mean_ptr + pid, mean_val)

@torch.fx.wrap
def fused_add_mean_3in(x0, x1, x2):
    """
    Fused kernel: (x0 + x1 + x2).mean(dim=[2,3], keepdim=True)
    Optimizes by:
    1. Fusing add and mean into single kernel
    2. Avoiding intermediate tensor allocation
    """
    n, c, h, w = x0.shape
    
    # Allocate output tensors
    out = torch.empty_like(x0)
    mean_out = torch.empty((n, c, 1, 1), dtype=x0.dtype, device=x0.device)
    
    # Launch kernel - one program per channel
    grid = (n * c,)
    BLOCK_SIZE = 1024
    hw_size = h * w
    
    fused_add_mean_kernel_3in[grid](
        x0, x1, x2,
        out, mean_out,
        n * c * h * w,
        hw_size,
        BLOCK_SIZE,
    )
    
    return out, mean_out


def pattern_in0_in1(in_0, in_1):
    """
    Match pattern: in_0 + in_1 followed by mean((2,3), keepdim=True)
    For graphs like: tmp_0 = 0 + in_1; tmp_0 += in_0; tmp_2 = tmp_1.mean((2,3), keepdim=True)
    """
    tmp_0 = in_0 + in_1
    tmp_1 = tmp_0
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return tmp_1, tmp_2


def replacement_args_in0_in1(in_0, in_1):
    return (in_0, in_1)


def replacement_func_in0_in1():
    return fused_add_mean_2in


def pattern_in1_in2_in0(in_0, in_1, in_2):
    """
    Match pattern: (in_1 + in_2 + in_0) followed by mean((2,3), keepdim=True)
    Pattern from graphs like repvgg: tmp_0 = in_1 + in_2; tmp_0 += in_0; tmp_2 = tmp_1.mean((2,3), keepdim=True)
    """
    tmp_0 = in_1 + in_2
    tmp_0 += in_0
    tmp_1 = tmp_0
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return tmp_1, tmp_2


def replacement_args_in1_in2_in0(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func_in1_in2_in0():
    return fused_add_mean_3in


def pattern_in0_in1_in2(in_0, in_1, in_2):
    """
    Match pattern: (in_0 + in_1 + in_2) followed by mean((2,3), keepdim=True)
    Pattern from graphs: tmp_0 = in_0 + in_1; tmp_0 += in_2; tmp_2 = tmp_1.mean((2,3), keepdim=True)
    """
    tmp_0 = in_0 + in_1
    tmp_0 += in_2
    tmp_1 = tmp_0
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return tmp_1, tmp_2


def replacement_args_in0_in1_in2(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func_in0_in1_in2():
    return fused_add_mean_3in


# Export all patterns/replacement_args/replacement_func as pattern, replacement_args, replacement_func
# The framework will discover them by suffix
pattern = pattern_in1_in2_in0
replacement_args = replacement_args_in1_in2_in0
replacement_func = replacement_func_in1_in2_in0