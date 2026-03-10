import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def se_block_fused_kernel(
    # Input pointers
    x_ptr,          # in_6: main input [B, C, H, W]
    x_se_ptr,       # in_7: SE input [B, C_se, 1, 1]
    weight_ptr,     # in_5: fc2 weight [C, C_se, 1, 1]
    bias_ptr,       # in_4: fc2 bias [C]
    # Batch norm params
    bn_mean_ptr,    # in_0: running_mean [C]
    bn_var_ptr,     # in_1: running_var [C]
    bn_bias_ptr,    # in_2: bias [C]
    bn_weight_ptr,  # in_3: weight [C]
    # Output pointers
    relu_out_ptr,   # tmp_9: output after relu
    bn_out_ptr,     # tmp_10: output after batch_norm
    # Sizes
    B, C, H, W, C_se,
    # Strides
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_xse_b, stride_xse_c,
    stride_w_c, stride_w_cse,
    # Meta
    M, N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused SE Block kernel that combines:
    1. conv2d (1x1): x_se @ weight + bias -> [B, C, 1, 1]
    2. sigmoid
    3. element-wise multiply: x * sigmoid_out
    4. relu
    5. batch_norm
    
    This kernel processes each (b, h, w) position independently,
    applying the SE block transformation.
    """
    # Grid: (B * H * W,)
    pid = tl.program_id(0)
    
    # Calculate batch and spatial coordinates
    num_spots = H * W
    b = pid // num_spots
    hw = pid % num_spots
    h = hw // W
    w = hw % W
    
    # Calculate offsets for main input x [B, C, H, W]
    x_base = b * stride_xb + h * stride_xh + w * stride_xw
    
    # Pointers for batch norm parameters
    bn_mean = tl.load(bn_mean_ptr)
    bn_var = tl.load(bn_var_ptr)
    bn_bias = tl.load(bn_bias_ptr)
    bn_weight = tl.load(bn_weight_ptr)
    
    # Compute batch_norm stats
    bn_std = tl.sqrt(bn_var + 1e-05)
    
    # Process each channel
    relu_out_acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    bn_out_acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    # Process in blocks
    for ch_start in range(0, C, BLOCK_SIZE_N):
        ch_offsets = ch_start + tl.arange(0, BLOCK_SIZE_N)
        ch_mask = ch_offsets < C
        
        # Load x [B, C, H, W] for this channel block
        x_ptrs = x_base + ch_offsets * stride_xc
        x_vals = tl.load(x_ptrs, mask=ch_mask, other=0.0)
        
        # Compute SE branch: conv2d(x_se, weight, bias) -> sigmoid -> multiply with x
        # x_se is [B, C_se, 1, 1], weight is [C, C_se, 1, 1]
        # This is essentially: for each output channel c, sum over c_se: x_se[b, c_se] * weight[c, c_se] + bias[c]
        
        # Load x_se [B, C_se, 1, 1] - same for all channels
        x_se_val = tl.load(x_se_ptr + b * stride_xse_b)
        
        # Compute conv2d: for each output channel, compute dot product
        # weight shape: [C, C_se, 1, 1], x_se shape: [B, C_se, 1, 1]
        # We need to compute: sum_cse(weight[c, c_se] * x_se[b, c_se]) + bias[c]
        
        se_out = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        for cse_start in range(0, C_se, BLOCK_SIZE_M):
            cse_offsets = cse_start + tl.arange(0, BLOCK_SIZE_M)
            cse_mask = cse_offsets < C_se
            
            # Load weight [C, C_se] for this channel block
            # weight is stored as [C, C_se, 1, 1], we need weight[c, c_se]
            w_ptrs = (ch_offsets[:, None] * stride_w_c + 
                      cse_offsets[None, :] * stride_w_cse)
            w_vals = tl.load(w_ptrs, mask=ch_mask[:, None] & cse_mask[None, :], other=0.0)
            
            # x_se is [B, C_se], load for these c_se
            x_se_block = tl.load(x_se_ptr + b * stride_xse_b + cse_offsets * stride_xse_c,
                                 mask=cse_mask, other=0.0)
            
            # Multiply and accumulate
            se_out += tl.sum(w_vals * x_se_block, axis=1)
        
        # Add bias
        bias_vals = tl.load(bias_ptr + ch_offsets, mask=ch_mask, other=0.0)
        se_out = se_out + bias_vals
        
        # Sigmoid
        sig_out = 1.0 / (1.0 + tl.exp(-se_out))
        
        # Multiply: x * sigmoid
        mult_out = x_vals * sig_out
        
        # Relu: max(0, mult_out)
        relu_out = tl.where(mult_out > 0, mult_out, 0.0)
        
        # Store relu output
        relu_ptrs = ch_offsets * stride_xc + b * stride_xb + h * stride_xh + w * stride_xw
        tl.store(relu_out_ptr + relu_ptrs, relu_out, mask=ch_mask)
        
        # Batch norm: (relu_out - mean) / std * weight + bias
        bn_out = (relu_out - bn_mean) / bn_std * bn_weight + bn_bias
        
        # Store batch_norm output
        bn_ptrs = ch_offsets * stride_xc + b * stride_xb + h * stride_xh + w * stride_xw
        tl.store(bn_out_ptr + bn_ptrs, bn_out, mask=ch_mask)


@torch.fx.wrap
def se_block_fused(
    x,           # in_6: main input [B, C, H, W]
    x_se,        # in_7: SE input [B, C_se, 1, 1] 
    weight,      # in_5: fc2 weight [C, C_se, 1, 1]
    bias,        # in_4: fc2 bias [C]
    bn_mean,     # in_0: running_mean [C]
    bn_var,      # in_1: running_var [C]
    bn_bias,     # in_2: bias [C]
    bn_weight,   # in_3: weight [C]
):
    """
    Fused SE Block implementation.
    
    Combines: conv2d -> sigmoid -> multiply -> relu -> batch_norm
    """
    B, C, H, W = x.shape
    C_se = x_se.shape[1]
    
    # Allocate outputs
    relu_out = torch.empty_like(x)
    bn_out = torch.empty_like(x)
    
    # Define block size
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 256
    
    # Calculate grid
    M = B * H * W
    N = C
    num_programs_m = M
    num_programs_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    se_block_fused_kernel[(num_programs_m,)](
        x, x_se, weight, bias,
        bn_mean, bn_var, bn_bias, bn_weight,
        relu_out, bn_out,
        B, C, H, W, C_se,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        x_se.stride(0), x_se.stride(1),
        weight.stride(0), weight.stride(1),
        M, N,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
    
    return relu_out, bn_out


def pattern(a, b, c, d, e, f, g, h):
    """
    Match the SE block pattern from model.py
    """
    t = a
    u = b
    v = c
    w = d
    x = e
    y = f
    z = torch.conv2d(h, y, x, (1, 1), (0, 0), (1, 1), 1)
    sig = z.sigmoid()
    mult = g * sig
    relu = torch.nn.functional.relu(mult, inplace=True)
    bn = torch.nn.functional.batch_norm(relu, t, u, w, v, False, 0.1, 1e-05)
    return (relu, bn)


def replacement_args(a, b, c, d, e, f, g, h):
    return (a, b, c, d, e, f, g, h)


def replacement_func():
    pass