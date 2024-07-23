import jax
from functools import partial


def get_kernel_ntk(kernel_fn):
    return jax.jit(partial(kernel_fn, get='ntk'))

