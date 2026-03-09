# Disabled pass - contains blocked APIs

def pattern(*args):
    # Disabled
    pass

def replacement_args(*args):
    return ()

def replacement_func():
    # Disabled replacement
    def disabled(*args):
        pass
    return disabled