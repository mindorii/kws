# End-to-end Keyword Spotting and Voice Activity Detection

The reference implementation for [An End-to-End Architecture for Keyword Spotting and Voice Activity Detection](https://arxiv.org/abs/1611.09405)

## Abstract

We propose a single neural network architecture for two tasks: on-line keyword spotting and voice activity detection. We develop novel inference algorithms for an end-to-end Recurrent Neural Network trained with the Connectionist Temporal Classification loss function which allow our model to achieve high accuracy on both keyword spotting and voice activity detection without retraining. In contrast to prior voice activity detection models, our architecture does not require aligned training data and uses the same parameters as the keyword spotting model. This allows us to deploy a high quality voice activity detector with no additional memory or maintenance requirements.

## Requirements

## Install

## Train

## Inference

## Citation

If you find this code useful for your research, please cite:
```
@inproceedings{LengerichHannun2016,
  author    = {Christopher T. Lengerich and Awni Y. Hannun},
  title     = {An End-to-End Architecture for Keyword Spotting and Voice Activity Detection},
  booktitle = {NIPS End-to-End Learning for Speech and Audio Processing Workshop},
  year      = {2016},
}