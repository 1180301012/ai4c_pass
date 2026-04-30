from pass_dir.shared_dispatch import dispatch


def pattern(scalar, vector):
    e = scalar.exp()
    result = e * vector
    return result


def replacement_args(scalar, vector):
    return ("exp_mul", scalar, vector)


def replacement_func():
    return dispatch