import torch
import triton
import triton.language as tl

@triton.jit
def fused_conv_cat_sigmoid_kernel(
    input_ptr, weight_ptr, bias_ptr,
    cat0_ptr, cat1_ptr, output_ptr,
    N, C_in, H, W,
    cat0_len, cat1_len, conv_len,
    total_elements, row_len,
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,
    stride_wt_ic,
    stride_cat0_n, stride_cat0_2,
    stride_cat1_n, stride_cat1_2,
    BLOCK_SIZE: tl.constexpr,
    REDUCE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    n_idx = offsets // row_len
    pos_in_row = offsets % row_len

    PI = 3.141592653589793

    # Case 1: from cat0 (in_3)
    is_cat0 = pos_in_row < cat0_len
    cat0_off = n_idx * stride_cat0_n + pos_in_row * stride_cat0_2
    cat0_val = tl.load(cat0_ptr + cat0_off, mask=mask & is_cat0, other=0.0).to(tl.float32)

    # Case 2: from cat1 (in_4)
    cat1_start = cat0_len
    cat1_end = cat0_len + cat1_len
    is_cat1 = (pos_in_row >= cat1_start) & (pos_in_row < cat1_end)
    cat1_local = pos_in_row - cat1_start
    cat1_off = n_idx * stride_cat1_n + cat1_local * stride_cat1_2
    cat1_val = tl.load(cat1_ptr + cat1_off, mask=mask & is_cat1, other=0.0).to(tl.float32)

    # Case 3: from conv2d (computed on-the-fly)
    is_conv = pos_in_row >= cat1_end
    conv_local = pos_in_row - cat1_end
    h_idx = conv_local // W
    w_idx = conv_local % W

    # Compute 1x1 conv: reduction over C_in channels
    # Initialize conv_val as block of zeros (to maintain consistent type in loop)
    conv_val = tl.zeros([BLOCK_SIZE], tl.float32)
    for c in range(REDUCE_SIZE):
        w_val = tl.load(weight_ptr + c * stride_wt_ic).to(tl.float32)
        in_off = n_idx * stride_in_n + c * stride_in_c + h_idx * stride_in_h + w_idx * stride_in_w
        in_val = tl.load(input_ptr + in_off, mask=mask & is_conv, other=0.0).to(tl.float32)
        conv_val = conv_val + w_val * in_val

    # Add bias after the loop (broadcast scalar to block)
    bias_val = tl.load(bias_ptr).to(tl.float32)
    conv_val = conv_val + bias_val

    # Select value based on source
    raw_val = tl.where(is_cat0, cat0_val, tl.where(is_cat1, cat1_val, conv_val))

    # Apply (sigmoid(x) - 0.25) * PI
    result = (tl.sigmoid(raw_val) - 0.25) * PI

    tl.store(output_ptr + offsets, result.to(output_ptr.dtype.element_ty), mask=mask)


@torch.fx.wrap
def fused_conv_cat_sigmoid_dispatch(input_tensor, weight_tensor, bias_tensor, cat0_tensor, cat1_tensor, route):
    N = input_tensor.shape[0]
    C_in = input_tensor.shape[1]
    H = input_tensor.shape[2]
    W = input_tensor.shape[3]

    cat0_len = cat0_tensor.shape[2]
    cat1_len = cat1_tensor.shape[2]
    conv_len = H * W
    total_len = cat0_len + cat1_len + conv_len
    row_len = total_len
    total_elements = N * total_len

    output = torch.empty((N, 1, total_len), dtype=input_tensor.dtype, device=input_tensor.device)

    BLOCK_SIZE = 1024
    REDUCE_SIZE = 64  # C_in is always 64

    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    fused_conv_cat_sigmoid_kernel[(num_programs,)](
        input_ptr=input_tensor,
        weight_ptr=weight_tensor,
        bias_ptr=bias_tensor,
        cat0_ptr=cat0_tensor,
        cat1_ptr=cat1_tensor,
        output_ptr=output,
        N=N, C_in=C_in, H=H, W=W,
        cat0_len=cat0_len, cat1_len=cat1_len, conv_len=conv_len,
        total_elements=total_elements, row_len=row_len,
        stride_in_n=input_tensor.stride(0),
        stride_in_c=input_tensor.stride(1),
        stride_in_h=input_tensor.stride(2),
        stride_in_w=input_tensor.stride(3),
        stride_wt_ic=weight_tensor.stride(1),
        stride_cat0_n=cat0_tensor.stride(0),
        stride_cat0_2=cat0_tensor.stride(2),
        stride_cat1_n=cat1_tensor.stride(0),
        stride_cat1_2=cat1_tensor.stride(2),
        BLOCK_SIZE=BLOCK_SIZE,
        REDUCE_SIZE=REDUCE_SIZE,
    )

    return output