import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.softmax(in_1, dim=1)
    tmp_1 = tmp_0.reshape(8, -1)
    tmp_2 = tmp_1.view(8, -1, 1, 1)
    tmp_3 = tmp_2.view(8, 2, -1, 1, 1)
    tmp_4 = tmp_3 * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    return tmp_0, tmp_5

def replacement_args(tmp_0, tmp_5):
    return (tmp_0, in_0)

@triton.jit
def fused_mul_sum_kernel(in0_ptr, softmax_ptr, out_ptr,
                         batch_size, height, width, channels,
                         in0_stride0, in0_stride1, in0_stride2, in0_stride3, in0_stride4,
                         softmax_stride0, softmax_stride1, softmax_stride2, softmax_stride3,
                         out_stride0, out_stride1, out_stride2, out_stride3):
    # Each thread handles one output element
    pid = tl.program_id(0)
    batch_id = pid // (height * width * channels)
    h_id = (pid // (width * channels)) % height
    w_id = (pid // channels) % width
    c_id = pid % channels

    # Unroll channel dimension (C=2 is fixed)
    total = tl.zeros((1,), dtype=tl.float32)
    for c_i in range(2):
        # Load input tensor element
        in0_val = tl.load(
            in0_ptr + batch_id * in0_stride0 + c_i * in0_stride1 + h_id * in0_stride2 + w_id * in0_stride3 + c_id * in0_stride4
        )
        # Load softmax element (shape [B, C, 1, D] → D is height)
        softmax_val = tl.load(
            softmax_ptr + batch_id * softmax_stride0 + c_i * softmax_stride1 + h_id * softmax_stride3
        )
        total = total + in0_val * softmax_val

    tl.store(
        out_ptr + batch_id * out_stride0 + h_id * out_stride1 + w_id * out_stride2 + c_id * out_stride3,
        total
    )

@torch.fx.wrap
def fused_mul_sum(in_0, in_1):
    # Compute softmax (external to Triton kernel)
    softmax_res = torch.nn.functional.softmax(in_1, dim=1)

    batch_size = softmax_res.shape[0]
    height = in_0.shape[2]  # Spatial height from in_0
    width = in_0.shape[3]   # Spatial width from in_0
    channels_out = in_0.shape[4]  # Output channels

    # Prepare output tensor
    out = torch.empty_like(in_0)

    # Get strides
    in0_strides = in_0.stride()
    softmax_strides = softmax_res.stride()
    out_strides = out.stride()

    # Grid setup: 1 thread per output element
    grid = (batch_size * height * width * channels_out,)

    # Launch Triton kernel
    fused_mul_sum_kernel[grid](
        in_0, softmax_res, out,
        batch_size, height, width, channels_out,
        *in0_strides,
        *softmax_strides,
        *out_strides
    )

    return out

def replacement_func():
    return fused_mul_sum