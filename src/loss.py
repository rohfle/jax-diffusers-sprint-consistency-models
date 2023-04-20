import jax.numpy as jnp


def l2_loss(logit, target):
    return (logit - target)**2

def l1_loss(logit, target):
    return jnp.abs(logit - target)

def get_loss_function(config):
    if config.training.loss_type == 'l1' :
        loss_fn = l1_loss
    elif config.training.loss_type == 'l2':
        loss_fn = l2_loss
    elif config.training.loss_type == 'mse':
        loss_fn = mse_loss
    else:
        raise NotImplementedError(
           f'loss_type {config.training.loss_tyoe} not supported yet!')

    return loss_fn