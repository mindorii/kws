# An End-to-End Architecture for Keyword Spotting and Voice Activity Detection

The reference implementation of the high quality keyword spotter introduced in [An End-to-End Architecture for Keyword
Spotting and Voice Activity Detection](https://arxiv.org/abs/1611.09405).

## Abstract

We propose a single neural network architecture for two tasks: on-line keyword
spotting and voice activity detection. We develop novel inference algorithms
for an end-to-end Recurrent Neural Network trained with the Connectionist
Temporal Classification loss function which allow our model to achieve high
accuracy on both keyword spotting and voice activity detection without
retraining. In contrast to prior voice activity detection models, our
architecture does not require aligned training data and uses the same
parameters as the keyword spotting model. This allows us to deploy a high
quality voice activity detector with no additional memory or maintenance
requirements.

[[Arxiv]](https://arxiv.org/abs/1611.09405)

## Requirements
This code has been run on Ubuntu 14.04 with Python 2.7 and Tensorflow 1.4.0.

We also require Boost Python. The build process will attempt to build the
Ubuntu package `libsamplerate-dev`. For this you may need root access or you
will need to modify the Makefiles.

## Install

### virtualenv

Setup the virtual environment:

```
virtualenv kws
source kws/bin/activate
```

### Dependencies

To install the system and python dependencies, from the repo root directory simply run:

```
make .deps
```

### Build

After the dependencies are installed, run:

```
make
```

## Data

[Dataset](https://drive.google.com/file/d/1wjJ7PYEJ8zFCoO6IEYaJyhxT266V1TKt/view?usp=sharing)

Along with code, we also provide a dataset of positive samples for the keyword "Olivia." To train a high-quality model as described in the paper, you'll need your own corpus of LV speech data and noise for negative samples. The dataset is split as follows:

* `train` (1544 samples)
* `test` (550 samples)

We also provide a dataset with added noise, as described in the paper:

* `train_noise` (15440 samples)

Download the data and unzip it into the `data` subdirectory.

## Train

Before training, make sure to set your `PYTHONPATH` to the repo top level
directory. From the repo top level directory run
```
export PYTHONPATH=$PYTHONPATH:`pwd`
```

All the configuration parameters can be found in `config/kws.json`. The most
important thing to change here are the paths to the data json(s) and where to
save the model. These are `config["data"]["train_jsons"]` and
`config["io"]["save_path"]` respectively.

Most of the other parameters should work out of the box, however you are free
to change these for hyperparameter tuning etc.

To train a model run
```
python train.py
```

For help / usage run

```
python train.py -h
```

This should produce:

```
usage: train.py [-h] [--config CONFIG] [--num_gpus NUM_GPUS]

optional arguments:
  -h, --help           show this help message and exit
  --config CONFIG      Configuration json for model building and training
  --num_gpus NUM_GPUS  Number of GPUs to train with.
```

## Evaluation

To evaluate a trained model use the `kws_eval.py` script. This takes as input a
model directory and a list of wave files. The wave files can be arbitrary
length, the model streams the evaluation.

```
usage: kws_eval.py [-h] --save_path SAVE_PATH --file_list FILE_LIST

optional arguments:
  -h, --help            show this help message and exit
  --save_path SAVE_PATH
                        Path where model is saved.
  --file_list FILE_LIST
                        Path to list of wave files.
```

## Citation

If you find this code useful for your research, please cite:
```
@inproceedings{LengerichHannun2016,
  author    = {Christopher T. Lengerich and Awni Y. Hannun},
  title     = {An End-to-End Architecture for Keyword Spotting and Voice Activity Detection},
  booktitle = {NIPS End-to-End Learning for Speech and Audio Processing Workshop},
  year      = {2016},
}
