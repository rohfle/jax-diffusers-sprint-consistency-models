# README

implementing consistency training and distillation in JAX.


### Running locally

```shell
python src/main.py  --mode=train --workdir=./workdir --config=src/configs/fashion_mnist.py
```

## Based on

- yiyixu [denoising-diffusion-flax](https://github.com/yiyixuxu/denoising-diffusion-flax/)
- ilovescience [consistency.ipynb](https://github.com/fastai/course22p2/blob/master/nbs/consistency.ipynb)
- openai [consistency_models](https://github.com/openai/consistency_models)
- the paper [https://arxiv.org/pdf/2303.01469.pdf](https://arxiv.org/pdf/2303.01469.pdf)

## License

Apache 2.0