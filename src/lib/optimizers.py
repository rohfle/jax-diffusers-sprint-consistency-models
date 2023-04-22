import optax


def create_optimizer(config):
    if config.optimizer == 'Adam':
        optimizer = optax.adam(
            learning_rate=config.lr,
        )
    elif config.optimizer == 'AdamW':
        optimizer = optax.adamw(
            learning_rate=config.lr,
        )
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer