import jax.numpy as jnp


def l2_loss(logit, target):
    return (logit - target)**2

def l1_loss(logit, target):
    return jnp.abs(logit - target)

def mse_loss(logit, target):
    return jnp.mean((logit - target) ** 2)

def get_loss_function(loss_type):
    if loss_type == 'l1' :
        loss_fn = l1_loss
    elif loss_type == 'l2':
        loss_fn = l2_loss
    elif loss_type == 'mse':
        loss_fn = mse_loss
    else:
        raise NotImplementedError(
           f'loss_type {loss_type} not supported yet!')

    return loss_fn