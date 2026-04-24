import torch
from pass_dir.shared_dispatch import dispatch


# ---------------------------------------------------------------------------
# Pattern: batch_norm(x, running_mean, running_var, weight, bias,
#                    False, 0.1, 1e-05)   -- inference mode
# ---------------------------------------------------------------------------
def pattern(x, running_mean, running_var, weight, bias):
    result = torch.nn.functional.batch_norm(
        x, running_mean, running_var, weight, bias, False, 0.1, 1e-05
    )
    return result


def replacement_args(x, running_mean, running_var, weight, bias):
    # Append route string so the shared dispatch wrapper knows which branch to run
    return (x, running_mean, running_var, weight, bias, "bn")


def replacement_func():
    return dispatch