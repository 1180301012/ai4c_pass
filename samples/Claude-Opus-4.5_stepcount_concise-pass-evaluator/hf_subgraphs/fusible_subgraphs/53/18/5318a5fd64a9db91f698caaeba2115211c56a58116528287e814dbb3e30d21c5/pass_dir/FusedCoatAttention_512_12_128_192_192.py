import torch
import triton
import triton.language as tl

# Pattern for graph: coat_lite_medium_384_start1025_end1034_65
# in_0: [1, 8, 64, 64], in_1: [1, 8, 145, 64], in_2: [1, 8, 145, 64]
# reshape to [1, 512, 12, 12], split [128, 192, 192]

def pattern(in_0, in_1, in_2):
    tmp_0 = in_1 @ in_0
    tmp_1 = in_1[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_2 = in_2[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_3 = tmp_2.transpose(-1, -2)
    tmp_4 = tmp_3.reshape(1, 512, 12, 12)
    tmp_5 = torch.functional.split(tmp_4, [128, 192, 192], dim=1)
    tmp_6 = tmp_5[0]
    tmp_7 = tmp_5[1]
    tmp_8 = tmp_5[2]
    return (tmp_0, tmp_6, tmp_7, tmp_8, tmp_1)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 512}),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_kernel_512_12(
    in_ptr,
    out0_ptr, out1_ptr, out2_ptr,
    n_elements,
    seq_plus_1: tl.constexpr,
    dim: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    n_heads: tl.constexpr,
    SPLIT0: tl.constexpr,
    SPLIT1: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    # Output position in reshaped tensor [1, n_heads*dim, H, W]
    c = offsets // (H * W)
    hw = offsets % (H * W)
    h = hw // W
    w = hw % W
    
    # Convert to input position
    head = c // dim
    feature = c % dim
    seq = h * W + w
    
    # Input index: in_2[0, head, seq+1, feature]
    in_idx = head * (seq_plus_1 * dim) + (seq + 1) * dim + feature
    val = tl.load(in_ptr + in_idx, mask=mask)
    
    # Determine which output tensor to write to based on channel
    is_out0 = c < SPLIT0
    is_out1 = (c >= SPLIT0) & (c < SPLIT0 + SPLIT1)
    is_out2 = c >= SPLIT0 + SPLIT1
    
    # Calculate output indices for each split
    out0_c = c
    out1_c = c - SPLIT0
    out2_c = c - SPLIT0 - SPLIT1
    
    out0_idx = out0_c * (H * W) + h * W + w
    out1_idx = out1_c * (H * W) + h * W + w
    out2_idx = out2_c * (H * W) + h * W + w
    
    tl.store(out0_ptr + out0_idx, val, mask=mask & is_out0)
    tl.store(out1_ptr + out1_idx, val, mask=mask & is_out1)
    tl.store(out2_ptr + out2_idx, val, mask=mask & is_out2)

@torch.fx.wrap
def optimized_forward(in_0, in_1, in_2):
    # Matmul
    tmp_0 = in_1 @ in_0
    # Slice of in_1
    tmp_1 = in_1[:, :, 1:, :]
    
    # Fused slice + transpose + reshape + split for in_2
    n_heads = 8
    dim = 64
    H = 12
    W = 12
    SPLIT0 = 128
    SPLIT1 = 192
    SPLIT2 = 192
    seq_plus_1 = in_2.shape[2]
    
    # Create three output tensors directly
    tmp_6 = torch.empty(1, SPLIT0, H, W, device=in_2.device, dtype=in_2.dtype)
    tmp_7 = torch.empty(1, SPLIT1, H, W, device=in_2.device, dtype=in_2.dtype)
    tmp_8 = torch.empty(1, SPLIT2, H, W, device=in_2.device, dtype=in_2.dtype)
    
    n_elements = n_heads * dim * H * W
    
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    fused_kernel_512_12[grid](
        in_2,
        tmp_6, tmp_7, tmp_8,
        n_elements=n_elements,
        seq_plus_1=seq_plus_1,
        dim=dim,
        H=H,
        W=W,
        n_heads=n_heads,
        SPLIT0=SPLIT0,
        SPLIT1=SPLIT1,
    )
    
    return (tmp_0, tmp_6, tmp_7, tmp_8, tmp_1)

def replacement_func():
    return optimized_forward