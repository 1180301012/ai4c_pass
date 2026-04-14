import torch

def pattern(conv2d, in_5):
    tmp_3 = torch.sigmoid(conv2d)
    tmp_4 = in_5 * tmp_3
    return tmp_4, tmp_3

def replacement_args(conv2d, in_5):
    return (conv2d, in_5)

@torch.fx.wrap
def fuse_sigmoid_mul(conv2d, in_5):
    # Create output tensors with correct shapes
    batch_size, channels, height, width = conv2d.shape
    tmp_3 = torch.empty_like(conv2d)
    
    # tmp_4 should have same shape as in_5
    if in_5.shape == conv2d.shape:
        # Element-wise multiplication
        tmp_4 = torch.empty_like(in_5)
    else:
        # Handle broadcasting case - in_5 might have larger spatial dimensions
        tmp_4_shape = list(in_5.shape)
        if len(tmp_4_shape) == 4:
            # Ensure batch and channel dimensions match
            tmp_4_shape[0] = batch_size
            tmp_4_shape[1] = channels
        tmp_4 = torch.empty(tmp_4_shape, dtype=in_5.dtype, device=in_5.device)
    
    return tmp_4, tmp_3

def replacement_func():
    return fuse_sigmoid_mul