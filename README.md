# Introduction

This is the official implementation of the paper "DIP: Dual Incongruity Perceiving Network for Sarcasm Detection", which is accepted by CVPR 2023.

## Abstract

Sarcasm indicates the literal meaning is contrary to the actual attitude. Considering the popularity and complementarity of image-text data, we investigate multi-modal sarcasm detection. Different from other multi-modal tasks, for the sarcastic data, there exists intrinsic incongruity between a pair of image and text as demonstrated in psychological theories. 

To tackle this issue, we propose a Dual Incongruity Perceiving (DIP) network consisting of two branches to mine the sarcastic information from factual and affective levels. For the factual aspect, we introduce a channel-wise reweighting strategy to obtain semantically discriminative embeddings, and leverage gaussian distribution to model the uncertain correlation caused by the incongruity. The distribution is generated from the latest data stored in the memory bank, which can adaptively model the difference of semantic similarity between sarcastic and non-sarcastic data. For the affective aspect, we utilize siamese layers with shared parameters to learn cross-modal sentiment information. Furthermore, we use the polarity value to construct a relation graph for the mini-batch, which forms the continuous contrastive loss to acquire affective embeddings. Extensive experiments demonstrate that our proposed method performs favorably against state-of-the-art approaches.

## Installation
Step 1: download data from ["Multi-Modal Sarcasm Detection in Twitter with Hierarchical Fusion Model"](https://github.com/ZLJ2015106/pytorch-multimodal_sarcasm_detection.git)

Step 2: Please install the following packages before running the code:   
&ensp;torch == 1.13.0   
&ensp;torchtext == 0.14.0   
&ensp;torchvision == 0.14.0   
&ensp;transformers == 4.23.1   
&ensp;tokenizers == 0.13.1   
&ensp;senticnet == 1.6

## Usage

```bash
python main.py
```
