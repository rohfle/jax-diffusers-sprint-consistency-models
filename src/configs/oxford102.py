import ml_collections

def get_config():

  config = ml_collections.ConfigDict()


  # wandb
  config.wandb = wandb = ml_collections.ConfigDict()
  wandb.entity = "hfjs-consistency"
  wandb.project = "hfjs-consistency-flower102"
  wandb.job_type = "training"
  wandb.name = None
  wandb.log_train = False
  wandb.log_sample = False
  wandb.log_model = False


  # training
  config.training = training = ml_collections.ConfigDict()
  training.num_epochs = 700000
  training.loss_type = 'mse'
  training.half_precision = True
  training.log_every_steps = 100
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
  data.dataset ='huggan/flowers-102-categories'
  data.batch_size = 64 * 4
  data.cache = False
  data.image_size = 128
  data.channels = 3
  data.shuffle_buffer_size = 10000
  data.use_streaming = True



  # model
  config.model = model = ml_collections.ConfigDict()
  model.dim = 64
  model.dim_mults = (1, 2, 4, 8)


  # optim
  config.optim = optim = ml_collections.ConfigDict()
  optim.optimizer = 'Adam'
  optim.lr = 3e-5


  config.seed = 42

  return config

