import torch
import triton
import triton.language as tl

# Pattern: interpolate(x, (64,64)) followed by sigmoid and multiply
# Branch A: interpolate(in_4, (64,64)) → sigmoid → in_3 * sigmoid_output
# This only matches the Branch A path (larger 64x64 tensor)

def pattern(x, y):
    tmp_3 = torch.nn.functional.interpolate(x, (64, 64), None, 'bilinear', False)
    tmp_4 = torch.sigmoid(tmp_3)
    tmp_5 = y * tmp_4
    return tmp_5

def replacement_args(x, y):
    return (x, y, "route_interp_sigmoid_mul")

# ---- Shared dispatch wrapper (same as in FuseSigmoidMul) ----

@triton.jit
def fused_interpolate_sigmoid_mul_kernel(
    x_ptr, y_ptr, out_ptr,
    N, C, H_in, W_in, H_out, W_out,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_h = tl.program_id(1) * BLOCK_H
    pid_w = tl.program_id(2) * BLOCK_W
    
    h_offsets = pid_h + tl.arange(0, BLOCK_H)
    w_offsets = pid_w + tl.arange(0, BLOCK_W)
    
    h_mask = h_offsets < H_out
    w_mask = w_offsets < W_out
    
    scale_h = H_in / H_out
    scale_w = W_in / W_out
    
    src_h_f = (h_offsets + 0.5) * scale_h - 0.5
    src_w_f = (w_offsets + 0.5) * scale_w - 0.5
    
    src_h_f = tl.maximum(src_h_f, 0.0)
    src_w_f = tl.maximum(src_w_f, 0.0)
    src_h_f = tl.minimum(src_h_f, H_in - 1.0)
    src_w_f = tl.minimum(src_w_f, W_in - 1.0)
    
    h0 = tl.floor(src_h_f).to(tl.int32)
    w0 = tl.floor(src_w_f).to(tl.int32)
    h1 = tl.minimum(h0 + 1, H_in - 1)
    w1 = tl.minimum(w0 + 1, W_in - 1)
    
    dh = src_h_f - h0.to(tl.float32)
    dw = src_w_f - w0.to(tl.float32)
    
    nc_offset = pid_nc * H_in * W_in
    
    v00 = tl.load(x_ptr + nc_offset + h0[:, None] * W_in + w0[None, :], 
                  mask=h_mask[:, None] & w_mask[None, :], other=0.0)
    v01 = tl.load(x_ptr + nc_offset + h0[:, None] * W_in + w1[None, :], 
                  mask=h_mask[:, None] & w_mask[None, :], other=0.0)
    v10 = tl.load(x_ptr + nc_offset + h1[:, None] * W_in + w0[None, :], 
                  mask=h_mask[:, None] & w_mask[None, :], other=0.0)
    v11 = tl.load(x_ptr + nc_offset + h1[:, None] * W_in + w1[None, :], 
                  mask=h_mask[:, None] & w_mask[None, :], other=0.0)
    
    top = v00 * (1.0 - dw[None, :]) + v01 * dw[None, :]
    bottom = v10 * (1.0 - dw[None, :]) + v11 * dw[None, :]
    interp_val = top * (1.0 - dh[:, None]) + bottom * dh[:, None]
    
    sig_val = tl.sigmoid(interp_val.to(tl.float32))
    
    out_nc_offset = pid_nc * H_out * W_out
    y_val = tl.load(y_ptr + out_nc_offset + h_offsets[:, None] * W_out + w_offsets[None, :], 
                    mask=h_mask[:, None] & w_mask[None, :], other=0.0)
    
    result = y_val.to(tl.float32) * sig_val
    
    tl.store(out_ptr + out_nc_offset + h_offsets[:, None] * W_out + w_offsets[None, :], 
             result, mask=h_mask[:, None] & w_mask[None, :])

@triton.jit
def fused_sigmoid_mul_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    sig_x = tl.sigmoid(x_f32)
    out_f32 = y_f32 * sig_x
    tl.store(out_ptr + offsets, out_f32, mask=mask)

def _fused_interpolate_sigmoid_mul_impl(x, y):
    N = y.shape[0]
    C = y.shape[1]
    H_in = x.shape[2]
    W_in = x.shape[3]
    H_out = 64
    W_out = 64
    
    out = torch.empty((N, C, H_out, W_out), dtype=y.dtype, device=y.device)
    
    BLOCK_H = 16
    BLOCK_W = 16
    
    grid = (
        N * C,
        triton.cdiv(H_out, BLOCK_H),
        triton.cdiv(W_out, BLOCK_W),
    )
    
    fused_interpolate_sigmoid_mul_kernel[grid](
        x_ptr=x, y_ptr=y, out_ptr=out,
        N=N, C=C, H_in=H_in, W_in=W_in, H_out=H_out, W_out=W_out,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
    )
    return out

def _fused_sigmoid_mul_impl(x, y):
    out = torch.empty_like(y)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_sigmoid_mul_kernel[grid](
        x_ptr=x, y_ptr=y, out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

@torch.fx.wrap
def dispatch_wrapper(*args):
    route = args[-1]
    actual_args = args[:-1]
    if route == "route_interp_sigmoid_mul":
        return _fused_interpolate_sigmoid_mul_impl(*actual_args)
    elif route == "route_sigmoid_mul":
        return _fused_sigmoid_mul_impl(*actual_args)
    else:
        raise ValueError(f"Unknown route: {route}")

def replacement_func():
    return dispatch_wrapper