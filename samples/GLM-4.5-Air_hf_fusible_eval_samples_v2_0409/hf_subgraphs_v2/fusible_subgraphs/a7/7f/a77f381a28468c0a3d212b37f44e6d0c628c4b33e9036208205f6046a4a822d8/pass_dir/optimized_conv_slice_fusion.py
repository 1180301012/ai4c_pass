import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1):
    """Match conv2d + channel slice pattern"""
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = conv2d[(slice(None, None, None), slice(None, 2048, None), slice(None, None, None), slice(None, None, None))]
    return (tmp_2, conv2d)

# Argument extraction function  
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized fused kernel
@triton.jit
def fused_conv_slice_kernel(
    input_ptr, weight_ptr, output_ptr,
    full_output_ptr,
    batch_size, in_c, in_h, in_w, out_c,
    slice_size, kh, kw,
    stride_h, stride_w, 
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel fusing conv2d and channel slicing"""
    pid = tl.program_id(0)
    
    if pid >= slice_size:
        return
        
    # Compute spatial output dimensions
    out_h = (in_h - kh) // stride_h + 1
    out_w = (in_w - kw) // stride_w + 1
    
    # Process each batch for this output channel
    for b in range(batch_size):
        acc = 0.0
        # Convolution computation
        for c_in in range(in_c):
            for kh_idx in range(kh):
                for kw_idx in range(kw):
                    # Input pixel location
                    in_h_pos = b * in_c * in_h * in_w + c_in * in_h * in_w + (kh_idx * stride_h) * in_w + (kw_idx * stride_w)
                    x_val = tl.load(input_ptr + in_h_pos)
                    
                    # Weight for this input channel and output channel
                    w_pos = pid * in_c + c_in
                    w_val = tl.load(weight_ptr + w_pos)
                    
                    acc += x_val * w_val
        
        # Store to sliced output
        out_slice_pos = b * slice_size * out_h * out_w + pid * out_h * out_w
        tl.store(output_ptr + out_slice_pos, acc)
        
        # Store to full output (if not sliced, store all channels)
        if True:  # Always compute full output for consistency
            for out_ch in range(out_c):
                acc_full = 0.0
                for c_in in range(in_c):
                    for kh_idx in range(kh):
                        for kw_idx in range(kw):
                            in_h_pos = b * in_c * in_h * in_w + c_in * in_h * in_w + (kh_idx * stride_h) * in_w + (kw_idx * stride_w)
                            x_val = tl.load(input_ptr + in_h_pos)
                            
                            w_pos_full = out_ch * in_c + c_in
                            w_val_full = tl.load(weight_ptr + w_pos_full)
                            
                            acc_full += x_val * w_val_full
                
                out_full_pos = b * out_c * out_h * out_w + out_ch * out_h * out_w
                tl.store(full_output_ptr + out_full_pos, acc_full)

@triton.jit  
def fused_conv_slice_2x2_kernel(
    input_ptr, weight_ptr, output_ptr,
    full_output_ptr,
    batch_size, in_c, in_h, in_w, out_c,
    slice_size, 
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for 2x2 conv2d + channel slicing"""
    pid = tl.program_id(0)
    
    if pid >= slice_size:
        return
        
    out_h = in_h // 2  # 2x2 stride 2 reduces spatial dims by half
    out_w = in_w // 2
    
    for b in range(batch_size):
        acc = 0.0
        # 2x2 convolution computation
        for c_in in range(in_c):
            for kh_idx in range(2):  # 2x2 kernel
                for kw_idx in range(2):
                    # Input pixel location with stride 2
                    in_h_pos = b * in_c * in_h * in_w + c_in * in_h * in_w + (kh_idx * 2) * in_w + (kw_idx * 2)
                    x_val = tl.load(input_ptr + in_h_pos)
                    
                    # Weight for this input channel and output channel  
                    w_pos = pid * in_c + c_in
                    w_val = tl.load(weight_ptr + w_pos)
                    
                    acc += x_val * w_val
        
        # Store to sliced output
        out_slice_pos = b * slice_size * out_h * out_w + pid * out_h * out_w
        tl.store(output_ptr + out_slice_pos, acc)
        
        # Store to full output
        for out_ch in range(out_c):
            acc_full = 0.0
            for c_in in range(in_c):
                for kh_idx in range(2):
                    for kw_idx in range(2):
                        in_h_pos = b * in_c * in_h * in_w + c_in * in_h * in_w + (kh_idx * 2) * in_w + (kw_idx * 2)
                        x_val = tl.load(input_ptr + in_h_pos)
                        
                        w_pos_full = out_ch * in_c + c_in
                        w_val_full = tl.load(weight_ptr + w_pos_full)
                        
                        acc_full += x_val * w_val_full
            
            out_full_pos = b * out_c * out_h * out_w + out_ch * out_h * out_w
            tl.store(full_output_ptr + out_full_pos, acc_full)

@torch.fx.wrap
def optimized_conv_slice_1x1(x, w, slice_size):
    """Optimized 1x1 conv with channel slicing"""
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels = w.shape[0]
    
    # Output spatial dimensions for 1x1 conv stride 1
    out_height, out_width = in_height, in_width
    
    # Create output tensors
    slice_output = torch.empty((batch_size, slice_size, out_height, out_width), 
                              dtype=x.dtype, device=x.device)
    full_output = torch.empty((batch_size, out_channels, out_height, out_width), 
                             dtype=x.dtype, device=x.device)
    
    # Launch kernel
    fused_conv_slice_kernel[(slice_size,)](
        x=x, w=w, 
        output_ptr=slice_output,
        full_output_ptr=full_output,
        batch_size=batch_size,
        in_c=in_channels,
        in_h=in_height,
        in_w=in_width,
        out_c=out_channels,
        slice_size=slice_size,
        kh=1, kw=1,  # 1x1 kernel
        stride_h=1, stride_w=1,
        BLOCK_SIZE=1024
    )
    
    return (slice_output, full_output)

@torch.fx.wrap
def optimized_conv_slice_2x2(x, w, slice_size):
    """Optimized 2x2 conv with channel slicing"""
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels = w.shape[0]
    
    # Output spatial dimensions for 2x2 conv stride 2
    out_height, out_width = in_height // 2, in_width // 2
    
    # Create output tensors
    slice_output = torch.empty((batch_size, slice_size, out_height, out_width), 
                              dtype=x.dtype, device=x.device)
    full_output = torch.empty((batch_size, out_channels, out_height, out_width), 
                             dtype=x.dtype, device=x.device)
    
    # Launch kernel
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

# Replacement function (returns optimized kernel function)
def replacement_func():
    # Return a function that selects the right kernel based on stride/dilation
    def optimized_conv_kernel(x, w, stride):
        stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride)
        
        # Determine kernel size from stride configuration
        kernel_size = 1 if stride_h == 1 and stride_w == 1 else 2
        
        # Extract slice size from the weight output channels (using a reasonable default)
        slice_size = min(2048, w.shape[0])  # Use 2048 as target slice size
        
        if kernel_size == 1:
            return optimized_conv_slice_1x1(x, w, slice_size)
        else:
            return optimized_conv_slice_2x2(x, w, slice_size)
    
    return optimized_conv_kernel