# TaylorGAN

## Install

Create a pipenv:

```bash
pipenv --three
```

Then, install the packages:

```bash
pipenv install
```

After installation, source it:

```bash
pipenv shell
```

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
train.py GAN [-h] --dataset {ai_challenger, chinese_poem, coco, coco_cleaned, news, news_cleaned, athleanx, athleanx_title}
             [--preprocessor {pipe, bert}] [--maxlen positive_int] [--vocab_size positive_int] [--cache_root_dir path]
             [-g GENERATOR_ID [k1=v1,k2=v2 ...]] [--g-fix-embeddings] [--tie-embeddings]
             [--g-optimizer OPTIMIZER_ID [k1=v1,k2=v2 ...]] [--g-regularizer REGULARIZER_ID [k1=v1,k2=v2 ...]]
             [-d DISCRIMINATOR_ID [k1=v1,k2=v2 ...]] [--d-fix-embeddings] [--sos]
             [--d-optimizer OPTIMIZER_ID [k1=v1,k2=v2 ...]] [--d-regularizer REGULARIZER_ID [k1=v1,k2=v2 ...]]
             [--estimator ESTIMATOR_ID [k1=v1,k2=v2 ...]] [--loss {alt, JS, KL, RKL}]
             [--iters-per-generator-update int] [--epochs int] [--batch-size int] [--random-seed int]
             [--bleu [int∈[1, 5]]] [--fed [positive_int]]
             [--checkpoint-root path] [--serving-root path] [--save-period int]
             [--tensorboard [path]] [--tags TAG [TAG ...]]
             [--jit] [--debug] [--profile [path]]
```

- IJCAI 2020 Parameters

```bash
train.py GAN --dataset news_cleaned \
             -g gru --tie-embeddings \
             --g-op adam learning_rate=1e-4,beta1=0.5,clip_global_norm=10 \
             --g-reg entropy coeff=0.02,discount=0 \
             -d cnn activation=elu \
             --d-op adam learning_rate=1e-4,beta1=0.5,clip_global_norm=10 \
             --d-reg spectral coeff=0.07 \
             --d-reg embedding coeff=0.2,max_norm=1 \
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
