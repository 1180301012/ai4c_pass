import torch

# Pattern matching: fuse element-wise add + dropout2d with training=False
# training=False makes dropout2d a no-op, so we just need the add operation
def pattern(in_3, in_4):
    # Element-wise addition
    tmp_3 = in_4 + in_3
    # Dropout2d with training=False is a no-op
    tmp_4 = torch.nn.functional.dropout2d(tmp_3, 0.1, False, False)
    return tmp_4

# Extract arguments from matched nodes
def replacement_args(in_3, in_4):
    return (in_3, in_4)

# Simple replacement using PyTorch's optimized add
# This removes the dropout2d call overhead while using PyTorch's well-optimized add
def replacement_func():
    def add_without_dropout(x, y):
        return x + y
    return add_without_dropout