import torch
import triton
import triton.language as tl

def pattern(x):
    """Match interpolate followed by sigmoid"""
    tmp = torch.nn.functional.interpolate(x, size=(640, 640), mode='bilinear')
    out = torch.nn.functional.sigmoid(tmp)
    return out

def replacement_args(x):
    return (x,)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['num_elements'],
)
@triton.jit
def bilinear_interp_sigmoid_kernel(
    x_ptr,
    out_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Convert linear index to (b, c, h_out, w_out)
    w_out = offsets % W_out
    h_out = (offsets // W_out) % H_out
    c = (offsets // (W_out * H_out)) % C
    b = offsets // (C * H_out * W_out)
    
    # Compute source coordinates for bilinear interpolation (align_corners=False)
    scale_h = H_in / H_out
    scale_w = W_in / W_out
    
    h_in_f = (h_out.to(tl.float32) + 0.5) * scale_h - 0.5
    w_in_f = (w_out.to(tl.float32) + 0.5) * scale_w - 0.5
    
    # Clamp to valid range
    h_in_f = tl.maximum(h_in_f, 0.0)
    w_in_f = tl.maximum(w_in_f, 0.0)
    h_in_f = tl.minimum(h_in_f, H_in - 1.0)
    w_in_f = tl.minimum(w_in_f, W_in - 1.0)
    
    # Get integer coordinates
    h0 = h_in_f.to(tl.int32)
    w0 = w_in_f.to(tl.int32)
    h1 = tl.minimum(h0 + 1, H_in - 1)
    w1 = tl.minimum(w0 + 1, W_in - 1)
    
    # Get interpolation weights
    h_weight = h_in_f - h0.to(tl.float32)
    w_weight = w_in_f - w0.to(tl.float32)
    
    # Compute input strides
    stride_b = C * H_in * W_in
    stride_c = H_in * W_in
    stride_h = W_in
    
    base = b * stride_b + c * stride_c
    
    idx00 = base + h0 * stride_h + w0
    idx01 = base + h0 * stride_h + w1
    idx10 = base + h1 * stride_h + w0
    idx11 = base + h1 * stride_h + w1
    
    # Load values
    v00 = tl.load(x_ptr + idx00, mask=mask, other=0.0)
    v01 = tl.load(x_ptr + idx01, mask=mask, other=0.0)
    v10 = tl.load(x_ptr + idx10, mask=mask, other=0.0)
    v11 = tl.load(x_ptr + idx11, mask=mask, other=0.0)
    
    # Bilinear interpolation
    w00 = (1.0 - h_weight) * (1.0 - w_weight)
    w01 = (1.0 - h_weight) * w_weight
    w10 = h_weight * (1.0 - w_weight)
    w11 = h_weight * w_weight
    
    interp = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11
    
    # Sigmoid
    result = tl.sigmoid(interp)
    
    # Store
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def bilinear_interp_sigmoid(x):
    B, C, H_in, W_in = x.shape
    H_out, W_out = 640, 640
    
    out = torch.empty((B, C, H_out, W_out), device=x.device, dtype=x.dtype)
    
    num_elements = B * C * H_out * W_out
    
    grid = lambda meta: ((num_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    bilinear_interp_sigmoid_kernel[grid](
        x, out,
        B, C, H_in, W_in,
        H_out, W_out,
        num_elements,
    )
    
    return out

def replacement_func():
    return bilinear_interp_sigmoid