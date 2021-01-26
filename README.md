# TaylorGAN

Source code of our NeurIPS 2020 poster paper *TaylorGAN: Neighbor-Augmented Policy Update Towards Sample-Efficient Natural Language Generation*

[Paper](https://neurips.cc/virtual/2020/public/poster_e1fc9c082df6cfff8cbcfff2b5a722ef.html) | [arXiv (including appendix)](https://arxiv.org/abs/2011.13527)

## Setup

### Environment

```bash
cp .env.sample .env
```

modify the `CHECKPOINT_DIR`, `DISK_CACHE_DIR`, `TENSORBOARD_PORT`, `TENSORBOARD_LOGDIR` as you need.

### Datasets

Download the text datasets from the following links:

[EMNLP 2017 News](https://github.com/pclucas14/GansFallingShort/tree/master/real_data_experiments/data/news)

[COCO Image Caption](https://github.com/pclucas14/GansFallingShort/tree/master/real_data_experiments/data/coco)

then, set the path in `datasets/corpus.yaml` to these text files.

### Pretrained Embeddings

Download the [pretrained fast text embeddings](https://drive.google.com/file/d/1w4AXIWYAukPbs7CQrH7_gxxpToSaecp1/view?usp=sharing)

and set the `PRETRAINED_EN_WORD_FASTTEXT_PATH` in `.env` to this file.

## Install

Install peotry first: [docs](https://python-poetry.org/docs/)

Install the packages:

```bash
$ poetry install
```

After installation:

```bash
$ poetry shell
```

### Tensorflow-GPU (**TODO**)

### For Developers (**TODO**)

## Scripts

### Train GAN

```bash
$ python src/scripts/train/GAN.py
```

- Usage

```bash
usage: GAN.py [-h] --dataset {coco_cleaned, news_cleaned, test} [--maxlen positive-int]
              [--vocab-size positive-int] [-g {gru, test}(*args, **kwargs)] [--tie-embeddings] [--g-fix-embeddings]
              [-d {cnn, resnet, test}(*args, **kwargs)] [--d-fix-embeddings]
              [--loss {alt, JS, KL, RKL}]
              [--estimator {reinforce, st, taylor, gumbel}(*args, **kwargs)]
              [--d-steps positive-int] [--g-regularizers REGULARIZER(*args, **kwargs) [REGULARIZER(*args, **kwargs) ...]]
              [--d-regularizers REGULARIZER(*args, **kwargs) [REGULARIZER(*args, **kwargs) ...]]
              [--g-optimizer {sgd, rmsprop, adam}(*args, **kwargs)]
              [--d-optimizer {sgd, rmsprop, adam}(*args, **kwargs)] [--epochs positive-int]
              [--batch-size positive-int] [--random-seed int] [--bleu [int∈[1, 5]]] [--fed [positive-int]] [--checkpoint-root Path]
              [--serving-root Path] [--save-period positive-int] [--tensorboard [Path]] [--tags TAG [TAG ...]] [--profile [Path]]

```

See more details and custom options for models/optimizers/regularizers:

```bash
python src/scripts/train/GAN.py -h
```

- NeurIPS 2020 Parameters

```bash
$ python src/scripts/train/GAN.py \
         --dataset news_cleaned \
         -g gru --tie-embeddings --g-reg 'entropy(0.02)' \
         -d 'cnn(activation="elu")' --d-reg 'spectral(0.07)' 'embedding(0.2, max_norm=1)' \
         --estimator 'taylor(bandwidth=0.5)' --loss RKL \
         --random-seed 2020 \
         --bleu 5 --fed 10000
```

## Run with Tensorboard

First, run a tensorboard,

```sh
sh src/scripts/run_tensorboard.sh
```

It will run a tensorboard that listen to `$TENSORBOARD_LOGDIR` and setup a server at port `$TENSORBOARD_PORT`. To change these settings, change these variables in `.env`.

Then, run the train script with tensorboard logging enabled:

```bash
python src/scripts/train/GAN.py ... --tensorboard
```

then, you can view the results by lauching `localhost:(TENSORBOARD_PORT)` with the web browser.

## Evaluate

**TODO**
