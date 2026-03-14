import torch
import triton
import triton.language as tl

# Pattern matching function - matches cat + view + layer_norm for 1536 case
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match the pattern: cat → view → layer_norm for 1536 dimension
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.cat([in_2, in_3, in_4, in_5], -1)
    tmp_3 = tmp_2.view(1, -1, 1536)
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (1536,), tmp_1, tmp_0, 1e-05)
    return tmp_4

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@triton.jit
def layer_norm_fwd_kernel(
    X0, X1, X2, X3,
    Y,
    W, B,
    M, N, stride,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_offset = row * stride
    offs = tl.arange(0, BLOCK_SIZE)
    mask0 = offs < stride
    
    # Load data
    x0 = tl.load(X0 + row_offset + offs, mask=mask0, other=0.0).to(tl.float32)
    x1 = tl.load(X1 + row_offset + offs, mask=mask0, other=0.0).to(tl.float32)
    x2 = tl.load(X2 + row_offset + offs, mask=mask0, other=0.0).to(tl.float32)
    x3 = tl.load(X3 + row_offset + offs, mask=mask0, other=0.0).to(tl.float32)
    
    # Mean
    sum_val = (tl.sum(tl.where(mask0, x0, 0.0), axis=0) + 
               tl.sum(tl.where(mask0, x1, 0.0), axis=0) +
               tl.sum(tl.where(mask0, x2, 0.0), axis=0) +
               tl.sum(tl.where(mask0, x3, 0.0), axis=0))
    mean = sum_val / N
    
    # Variance
    d0 = tl.where(mask0, x0 - mean, 0.0)
    d1 = tl.where(mask0, x1 - mean, 0.0)
    d2 = tl.where(mask0, x2 - mean, 0.0)
    d3 = tl.where(mask0, x3 - mean, 0.0)
    var = (tl.sum(d0*d0, axis=0) + tl.sum(d1*d1, axis=0) + 
           tl.sum(d2*d2, axis=0) + tl.sum(d3*d3, axis=0)) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Load weights/biases and normalize
    w0 = tl.load(W + offs, mask=mask0, other=1.0).to(tl.float32)
    w1 = tl.load(W + stride + offs, mask=mask0, other=1.0).to(tl.float32)
    w2 = tl.load(W + 2*stride + offs, mask=mask0, other=1.0).to(tl.float32)
    w3 = tl.load(W + 3*stride + offs, mask=mask0, other=1.0).to(tl.float32)
    b0 = tl.load(B + offs, mask=mask0, other=0.0).to(tl.float32)
    b1 = tl.load(B + stride + offs, mask=mask0, other=0.0).to(tl.float32)
    b2 = tl.load(B + 2*stride + offs, mask=mask0, other=0.0).to(tl.float32)
    b3 = tl.load(B + 3*stride + offs, mask=mask0, other=0.0).to(tl.float32)
    
    # Normalize and store
    out_row = row * N
    tl.store(Y + out_row + offs, (x0 - mean) * rstd * w0 + b0, mask=mask0)
    tl.store(Y + out_row + stride + offs, (x1 - mean) * rstd * w1 + b1, mask=mask0)
    tl.store(Y + out_row + 2*stride + offs, (x2 - mean) * rstd * w2 + b2, mask=mask0)
    tl.store(Y + out_row + 3*stride + offs, (x3 - mean) * rstd * w3 + b3, mask=mask0)


@torch.fx.wrap
def fused_cat_view_layernorm_1536(in_0, in_1, in_2, in_3, in_4, in_5):
    batch = in_2.shape[0]
    spatial_h = in_2.shape[1]
    spatial_w = in_2.shape[2]
    feature_dim = in_2.shape[3]
    
    M = batch * spatial_h * spatial_w
    N = 4 * feature_dim
    
    x0 = in_2.contiguous().view(-1, feature_dim)
    x1 = in_3.contiguous().view(-1, feature_dim)
    x2 = in_4.contiguous().view(-1, feature_dim)
    x3 = in_5.contiguous().view(-1, feature_dim)
    
    out = torch.empty((1, M, N), dtype=in_2.dtype, device=in_2.device)
    
    BLOCK_SIZE = 512
    grid = (M,)
    layer_norm_fwd_kernel[grid](
        x0, x1, x2, x3, out,
        in_1, in_0, M, N, feature_dim,
        eps=1e-05, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    
    return out


def replacement_func():
    return fused_cat_view_layernorm_1536