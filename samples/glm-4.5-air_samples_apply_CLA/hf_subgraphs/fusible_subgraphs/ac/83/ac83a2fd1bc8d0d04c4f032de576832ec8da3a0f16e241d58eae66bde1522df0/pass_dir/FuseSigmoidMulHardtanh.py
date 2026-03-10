import torch


# Pattern matching function - matches mul + hardtanh pattern
# The sigmoid is already computed before this pattern
def pattern(in_2, sigmoid_conv):
    # This pattern matches:
    # tmp_4 = in_2 * tmp_3      (element-wise multiplication)
    # tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    # Only return the final output that is used in the model return
    mul_out = in_2 * sigmoid_conv
    hardtanh_out = torch.nn.functional.hardtanh(mul_out, 0.0, 6.0, False)
    return hardtanh_out


# Extract arguments needed for replacement
def replacement_args(in_2, sigmoid_conv):
    return (in_2, sigmoid_conv)


def replacement_func():
    # Use PyTorch's native clamp which is highly optimized
    def fused_op(in_2, sigmoid_conv):
        # This fuses mul + clamp (hardtanh) into a single operation
        return (in_2 * sigmoid_conv).clamp(0.0, 6.0)
    return fused_op