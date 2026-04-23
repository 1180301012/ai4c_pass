import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_3, in_0, None)
    tmp = linear * in_1
    return (tmp, linear)

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_3)

# Optimized kernel
def fused_linear_scale_kernel(
    in_2_ptr,  # [batch, seq, in]
    in_0_ptr,  # [out, in]
    in_1_ptr,  # [out]
    out_ptr,   # [batch, seq, out]
    batch: tl.constexpr,
    seq: tl.constexpr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_IN: tl.constexpr
):
    # Block index for out dimension
    k_start = tl.program_id(0) * BLOCK_OUT
    k = k_start + tl.arange(0, BLOCK_OUT)
    k_mask = k < out_features

    i = tl.program_id(1)  # batch index
    j = tl.program_id(2)  # seq index

    # Load scale for current k block
    scale = tl.load(in_1_ptr + k, mask=k_mask)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)

    # Tile over input features
    for d_start in range(0, in_features, BLOCK_IN):
        d = d_start + tl.arange(0, BLOCK_IN)
        d_mask = d < in_features

        # Load input tensor
        x = tl.load(
            in_2_ptr + i * seq * in_features + j * in_features + d,
            mask=d_mask,
            other=0.0
        )

        # Load weights
        w = tl.load(
            in_0_ptr + k[:, None] * in_features + d,
            mask=tl.expand_dims(k_mask, 1) * d_mask,
            other=0.0
        )

        # Accumulate dot product
        acc += tl.sum(x * w, axis=0)

    # Apply scaling factor
    result = acc * scale
    # Store result
    tl.store(
        out_ptr + i * seq * out_features + j * out_features + k,
        result,
        mask=k_mask
    )


# Kernel wrapper
@torch.fx.wrap
def fused_linear_scale(in_0, in_1, in_2):
    batch, seq, in_features = in_2.shape
    out_features = in_0.shape[0]

    out = torch.empty_like(in_2)
    BLOCK_OUT = 128
    BLOCK_IN = 64

    grid = (
        triton.cdiv(out_features, BLOCK_OUT),
        batch,
        seq
    )

    # Launch kernel
    fused_linear_scale_kernel[grid](
        in_2_ptr=in_2,
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        batch=batch,
        seq=seq,
        in_features=in_features,
        out_features=out_features,
        BLOCK_OUT=BLOCK_OUT,
        BLOCK_IN=BLOCK_IN
    )

    return out

# Replacement function

def replacement_func():
    return fused_linear_scale