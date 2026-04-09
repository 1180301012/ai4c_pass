import torch
import triton
import triton.language as tl

# Pattern matching function for slice size 128
def pattern(in_0, in_1):
    """Match conv2d + channel slice pattern with slice size 128"""
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = conv2d[(slice(None, None, None), slice(None, 128, None), slice(None, None, None), slice(None, None, None))]
    return (tmp_2, conv2d)

# Argument extraction function  
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized fused kernel for 1x1 conv
@triton.jit
def fused_conv_slice_1x1_kernel(
    input_ptr, weight_ptr, output_ptr,
    full_output_ptr,
    batch_size, in_c, in_h, in_w, out_c,
    slice_size, BLOCK_SIZE: tl.constexpr,
):
    """Optimized 1x1 kernel fusing conv2d and channel slicing"""
    pid = tl.program_id(0)
    
    if pid >= slice_size:
        return
        
    out_h = in_h  # 1x1 conv preserves spatial dims
    out_w = in_w
    
    for b in range(batch_size):
        acc = 0.0
        # 1x1 convolution computation
        for c_in in range(in_c):
            # Load input value
            x_idx = b * in_c * in_h * in_w + c_in * in_h * in_w
            x_val = tl.load(input_ptr + x_idx)
            
            # Load weight
            w_idx = pid * in_c + c_in
            w_val = tl.load(weight_ptr + w_idx)
            
            acc += x_val * w_val
        
        # Store to sliced output
        out_slice_idx = b * slice_size * out_h * out_w + pid * out_h * out_w
        tl.store(output_ptr + out_slice_idx, acc)
        
        # Store to full output for all channels
        for out_ch in range(out_c):
            acc_full = 0.0
            for c_in in range(in_c):
                x_val = tl.load(input_ptr + b * in_c * in_h * in_w + c_in * in_h * in_w)
                w_val_full = tl.load(weight_ptr + out_ch * in_c + c_in)
                acc_full += x_val * w_val_full
            
            out_full_idx = b * out_c * out_h * out_w + out_ch * out_h * out_w
            tl.store(full_output_ptr + out_full_idx, acc_full)

# Optimized fused kernel for 2x2 conv with stride 2
@triton.jit
def fused_conv_slice_2x2_kernel(
    input_ptr, weight_ptr, output_ptr,
    full_output_ptr,
    batch_size, in_c, in_h, in_w, out_c,
    slice_size, BLOCK_SIZE: tl.constexpr,
):
    """Optimized 2x2 kernel fusing conv2d and channel slicing"""
    pid = tl.program_id(0)
    
    if pid >= slice_size:
        return
        
    out_h = in_h // 2  # 2x2 stride 2 reduces spatial dims
    out_w = in_w // 2
    
    for b in range(batch_size):
        acc = 0.0
        # 2x2 convolution computation  
        for c_in in range(in_c):
            # Load input value with stride 2
            x_idx = b * in_c * in_h * in_w + c_in * in_h * in_w
            x_val = tl.load(input_ptr + x_idx)
            
            # Load weight
            w_idx = pid * in_c + c_in
            w_val = tl.load(weight_ptr + w_idx)
            
            acc += x_val * w_val
        
        # Store to sliced output
        out_slice_idx = b * slice_size * out_h * out_w + pid * out_h * out_w
        tl.store(output_ptr + out_slice_idx, acc)
        
        # Store to full output for all channels
        for out_ch in range(out_c):
            acc_full = 0.0
            for c_in in range(in_c):
                x_val = tl.load(input_ptr + b * in_c * in_h * in_w + c_in * in_h * in_w)
                w_val_full = tl.load(weight_ptr + out_ch * in_c + c_in)
                acc_full += x_val * w_val_full
            
            out_full_idx = b * out_c * out_h * out_w + out_ch * out_h * out_w
            tl.store(full_output_ptr + out_full_idx, acc_full)

@torch.fx.wrap
def optimized_conv_slice_1x1(x, w, slice_size):
    """Optimized 1x1 conv with channel slicing"""
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels = w.shape[0]
    
    out_height, out_width = in_height, in_width
    
    slice_output = torch.empty((batch_size, slice_size, out_height, out_width), 
                              dtype=x.dtype, device=x.device)
    full_output = torch.empty((batch_size, out_channels, out_height, out_width), 
                             dtype=x.dtype, device=x.device)
    
    fused_conv_slice_1x1_kernel[(slice_size,)](
        x=x, w=w, 
        output_ptr=slice_output,
        full_output_ptr=full_output,
        batch_size=batch_size,
        in_c=in_channels,
        in_h=in_height,
        in_w=in_width,
        out_c=out_channels,
        slice_size=slice_size,
        BLOCK_SIZE=1024
    )
    
    return (slice_output, full_output)

@torch.fx.wrap
def optimized_conv_slice_2x2(x, w, slice_size):
    """Optimized 2x2 conv with channel slicing"""
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels = w.shape[0]
    
    out_height, out_width = in_height // 2, in_width // 2
    
    slice_output = torch.empty((batch_size, slice_size, out_height, out_width), 
                              dtype=x.dtype, device=x.device)
    full_output = torch.empty((batch_size, out_channels, out_height, out_width), 
                             dtype=x.dtype, device=x.device)
    
    fused_conv_slice_2x2_kernel[(slice_size,)](
        x=x, w=w,
        output_ptr=slice_output, 
        full_output_ptr=full_output,
        batch_size=batch_size,
        in_c=in_channels,
        in_h=in_height,
        in_w=in_width,
        out_c=out_channels,
        slice_size=slice_size,
        BLOCK_SIZE=1024
    )
    
    return (slice_output, full_output)

def replacement_func():
    """Return optimized kernel function"""
    def optimized_conv_kernel(x, w, stride):
        stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride)
        slice_size = 128  # Fixed slice size for this pass
        
        # Determine if this is 1x1 or 2x2 conv
        if stride_h == 1 and stride_w == 1:
            return optimized_conv_slice_1x1(x, w, slice_size)
        else:
            return optimized_conv_slice_2x2(x, w, slice_size)
    
    return optimized_conv_kernel