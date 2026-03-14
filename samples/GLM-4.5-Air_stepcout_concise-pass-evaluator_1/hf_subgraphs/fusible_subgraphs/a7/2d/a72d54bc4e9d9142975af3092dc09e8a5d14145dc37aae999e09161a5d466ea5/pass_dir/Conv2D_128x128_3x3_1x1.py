import torch
import triton
import triton.language as tl

def pattern(in_6, tmp_0):
    tmp_5 = torch.conv2d(in_6, tmp_0, None, (1, 1), (1, 1), (1, 1), 1)
    return tmp_5

def replacement_args(in_6, tmp_0):
    return (in_6, tmp_0)

@triton.jit
def conv2d_kernel(
    in_ptr, weight_ptr, out_ptr,
    N, C_out, C_in, H_in, W_in, H_out, W_out
):
    pid = tl.program_id(0)
    total_elements = N * C_out * H_out * W_out
    
    if pid >= total_elements:
        return
    
    in_y = pid // (C_out * H_out * W_out)
    in_x_c = (pid // (H_out * W_out)) % C_out
    in_y_out = (pid // W_out) % H_out
    in_x_out = pid % W_out
    
    result = 0.0
    
    for c_in in range(C_in):
        weight_offset = in_x_c * C_in * 9 + c_in * 9
        
        for kh in range(3):
            h_in = in_y_out + kh
            h_valid = (h_in >= 0)
            h_in_bounds = (h_in < H_in)
            
            for kw in range(3):
                w_in = in_x_out + kw
                w_valid = (w_in >= 0)
                w_in_bounds = (w_in < W_in)
                
                if (h_valid and h_in_bounds) and (w_valid and w_in_bounds):
                    in_idx = (in_y * C_in * H_in * W_in + 
                             c_in * H_in * W_in + 
                             h_in * W_in + w_in)
                    weight_idx = weight_offset + kh * 3 + kw
                    
                    in_val = tl.load(in_ptr + in_idx)
                    weight_val = tl.load(weight_ptr + weight_idx)
                    result += in_val * weight_val
    
    tl.store(out_ptr + pid, result)

@torch.fx.wrap
def optimized_conv2d(in_6, tmp_0):
    N, C_in, H_in, W_in = in_6.shape
    C_out, _, _, _ = tmp_0.shape
    
    H_out = (H_in + 2 * 1 - 3 * 1) // 1 + 1
    W_out = (W_in + 2 * 1 - 3 * 1) // 1 + 1
    
    out = torch.empty((N, C_out, H_out, W_out), device=in_6.device, dtype=in_6.dtype)
    
    total_elements = N * C_out * H_out * W_out
    
    conv2d_kernel[(total_elements,)](
        in_ptr=in_6,
        weight_ptr=tmp_0,
        out_ptr=out,
        N=N, C_out=C_out, C_in=C_in, H_in=H_in, W_in=W_in, H_out=H_out, W_out=W_out
    )
    
    return out

def replacement_func():
    return optimized_conv2d