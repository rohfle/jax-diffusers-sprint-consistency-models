import optax


def create_optimizer(config):
    if config.optimizer == 'Adam':
        optimizer = optax.adam(
            learning_rate=config.lr,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.eps,
        )
    elif config.optimizer == 'AdamW':
        optimizer = optax.adamw(
            learning_rate=config.lr,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.eps,
        )
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer