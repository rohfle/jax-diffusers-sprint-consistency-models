import ml_collections

def get_config():

  config = ml_collections.ConfigDict()


  # wandb
  config.wandb = wandb = ml_collections.ConfigDict()
  wandb.entity = "hfjs-consistency" # team name, must have already created
  wandb.project = "hfjs-consistency-fashion-mnist"  # required filed if use W&B logging
  wandb.job_type = "training"
  wandb.name = None # run name, optional
  wandb.log_train = False # log training metrics
  wandb.log_sample = False # log generated samples to W&B
  wandb.log_model = False # log final model checkpoint as W&B artifact


  # training
  config.training = training = ml_collections.ConfigDict()
  training.num_epochs = 10
  training.loss_type = 'mse'
  training.half_precision = True
  training.save_and_sample_every = 1000
  training.num_samples = 64
  training.epsilon = 0.002
  training.N = 150


  # ema
  config.ema = ema = ml_collections.ConfigDict()
  # TODO ?


  # ddpm
  config.ddpm = ddpm = ml_collections.ConfigDict()
  # TODO ?


  # data
  config.data = data = ml_collections.ConfigDict()
  data.batch_size = 128 * 8
  data.channels = 1
  data.dataset = 'fashion_mnist'
  data.image_size = 28
  data.shuffle_buffer_size = 10000
  data.use_streaming = True


  # model
  config.model = model = ml_collections.ConfigDict()
  model.dim = 64
  model.dim_mults = (1, 2, 4)


  # optim
  config.optim = optim = ml_collections.ConfigDict()
  optim.optimizer = 'AdamW'
  optim.lr = 3e-5

  config.seed = 42

  return config


