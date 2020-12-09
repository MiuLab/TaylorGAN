# TaylorGAN

Source code of our NeurIPS 2020 poster paper *TaylorGAN: Neighbor-Augmented Policy Update Towards Sample-Efficient Natural Language Generation*

[Paper](https://neurips.cc/virtual/2020/public/poster_e1fc9c082df6cfff8cbcfff2b5a722ef.html) | [arXiv (including appendix)](https://arxiv.org/abs/2011.13527)

## Setup

### Environment

```bash
cp .env.sample .env
```

modify the `DISK_CACHE_DIR`, `TENSORBOARD_PORT`, `TENSORBOARD_LOGDIR` as you need.

### Datasets

Download the corpus from the following links:

[EMNLP 2017 News](https://github.com/pclucas14/GansFallingShort/tree/master/real_data_experiments/data/news)
[COCO Image Caption](https://github.com/pclucas14/GansFallingShort/tree/master/real_data_experiments/data/coco)

then, set the path in `datasets/corpus.yaml` to these text files.

### Pretrained Embeddings

Download the pretrained fast text embeddings from **TODO Link**

and set the `PRETRAINED_EN_WORD_FASTTEXT_PATH` in `.env` to this file.

## Install

Create a pipenv:

```bash
pipenv --three
```

Then, install the packages:

```bash
pipenv install
```

### For Developers

**TODO** 

After installation

```bash
pipenv shell
```

Anytime you modify `.env`, you may reactivate the shell with the above to reload the variables.

## Scripts

```bash
python scripts/xxx.py
```

or

```bash
python -m scripts.xxx
```

use --help to get more details.

### Train GAN

- Usage

```bash
train.py [-h] --dataset {test} [--maxlen positive_int] [--vocab_size positive_int]
         [--loss {alt, JS, KL, RKL}] [--estimator ESTIMATOR_ID [k1=v1,k2=v2 ...]] [--d-steps int]
         [-g GENERATOR_ID [k1=v1,k2=v2 ...]] [--tie-embeddings] [--g-fix-embeddings] [--g-optimizer OPTIMIZER_ID [k1=v1,k2=v2 ...]]
         [--g-regularizer REGULARIZER_ID [k1=v1,k2=v2 ...]] [-d DISCRIMINATOR_ID [k1=v1,k2=v2 ...]] [--d-fix-embeddings]
         [--d-optimizer OPTIMIZER_ID [k1=v1,k2=v2 ...]] [--d-regularizer REGULARIZER_ID [k1=v1,k2=v2 ...]] [--epochs int] [--batch-size int]
         [--random-seed int] [--bleu [int∈[1, 5]]] [--fed [positive_int]] [--checkpoint-root path] [--serving-root path] [--save-period int]
         [--tensorboard [path]] [--tags TAG [TAG ...]] [--jit] [--debug] [--profile [path]]
```

- NeurIPS 2020 Parameters

```bash
train.py --dataset news_cleaned \
         -g gru --tie-embeddings \
         --g-op adam learning_rate=1e-4,beta1=0.5,clip_global_norm=10 \
         --g-reg entropy c=0.02,impl=dense \
         -d cnn activation=elu \
         --d-op adam learning_rate=1e-4,beta1=0.5,clip_global_norm=10 \
         --d-reg spectral c=0.07 \
         --d-reg embedding c=0.2,max_norm=1 \
         --estimator taylor bandwidth=0.5 \
         --loss RKL \
         --itgu 1 --random-seed 2020 \
         --bleu 5 --fed 10000
```

## Run with Tensorboard

First, run a tensorboard,

```sh
sh scripts/run_tensorboard.sh
```

It will run a tensorboard that listen to `$TENSORBOARD_LOGDIR` and setup a server at port `$TENSORBOARD_PORT`. To change these settings, change these variables in `.env`.

Then, run the train script with tensorboard logging enabled:

```bash
python -m scripts.train_GAN ... --tensorboard
```

## Evaluate

**TODO**
