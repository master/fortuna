import jax
import jax.numpy as jnp
import jax.scipy as jsp

from fortuna.typing import Array


def cross_entropy_loss_fn(
    outputs: Array,
    targets: Array,
) -> jnp.ndarray:
    """
    A cross-entropy loss function. Check `here <https://en.wikipedia.org/wiki/Cross_entropy>`_ for reference.

    Parameters
    ----------
    outputs: Array
        Model outputs to be passed to the loss.
    targets: Array
        Target data points.

    Returns
    -------
    jnp.ndarray
        The focal loss evaluation.
    """
    targets = jax.nn.one_hot(targets, outputs.shape[-1])
    return jnp.mean(jsp.special.logsumexp(outputs, -1) - jnp.sum(targets * outputs, -1))
