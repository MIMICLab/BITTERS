# Large-Scale Bidirectional Training for Zero-Shot Image Captioning

**Taehoon Kim<sup>1</sup>, Mark Marsden<sup>2</sup>, Pyunghwan Ahn<sup>1</sup>, Sangyun Kim<sup>1</sup>, Sihaeng Lee<sup>1</sup>, Alessandra Sala<sup>2</sup>, Seung Hwan Kim<sup>1</sup> [[Paper]](https://arxiv.org/abs/2211.06774.pdf)**

**1. LG AI Research**

**2. Shutterstock** 


<img src=assets/bitters.png width=1280>

## Abstract
When trained on large-scale datasets, image captioning models can understand the content of images from a general domain but often fail to generate accurate, detailed captions. To improve performance, pretraining-and-finetuning has been a key strategy for image captioning. However, we find that large-scale bidirectional training between image and text enables zero-shot image captioning. In this paper, we introduce Bidirectional Image Text Training in largER Scale, BITTERS, an efficient training and inference framework for zero-shot image captioning. We also propose a new evaluation benchmark which comprises of high quality datasets and an extensive set of metrics to properly evaluate zero-shot captioning accuracy and societal bias. We additionally provide an efficient finetuning approach for keyword extraction. We show that careful selection of large-scale training set and model architecture is the key to achieving zero-shot image captioning.

## Preparation

### Requirements

```
pip install -r requirements.txt
```

### Dataset

Place training and validation images in separate directories.


### Pretrained weights 

- We provide the WaveVAE pretrained weights on ImageNet dataset. 

    WaveNet: [Google Drive](https://drive.google.com/file/d/1gfB_vyw9hQr2MVyNFDCa4ssCLZ3Bn-M6/view?usp=share_link)

    WaveVAE: [Google Drive](https://drive.google.com/file/d/1PcQeJQUL8nGSGSzrUSxYm8Z4XmdKaSps/view?usp=share_link)

    WaveGAN: [Google Drive](https://drive.google.com/file/d/17sUKaCtdlwH3BGfLjS1MGSoCEV9PanYW/view?usp=share_link)

    WaveGAN is the final trained weight after Stage 2 training.


## WaveVAEs

<img src=assets/wavevae.png width=1280>

### Training

For faster training, our training code supports multi-gpu. 
To enable multi-gpu training, add " --gpus " flag with number of gpus in your machine (default 1).


For training, provide config file and training dataset.
Please refer to example config files in configs. 

imagenet_wnet.yaml: config for Stage 1 pretraining.
imagenet_wvae.yaml: config for calibration before Stage 2.
imagenet_wgan.yaml: config for Stage 2 training.

In this repository, WaveGAN is the final model that we used for BITTERS.

```
python train_vae.py --configs [config_file] --train_dir [path_to_train_data] --val_dir [path_to_val_data]
```

You can also test functionality with randomly generated fake data.

```
python train_vae.py --fake_data --configs [config_file] 
```

### Evaluation 

For faster evaluation, our evaluation code supports multi-gpu. 
To enable multi-gpu evaluation, add " --gpus " flag with number of gpus in your machine (default 1).

For evaluation, provide config file, pretrained WaveGAN weight, and test dataset
Please refer to example config files in configs. 


```
python eval_vae.py --configs [config_file] --ckpt_path [weight_file] --test_dir [path_to_test_data] 
```

You can also test functionality with randomly generated fake data.
```
python eval_vae.py --fake_data --configs [config_file] --ckpt_path [weight_file]
```

## BiART

Among many open-sourced Transformer (GPT) repositories, we used Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT) with extra embedding layer for Segment Embedding. 

Here's an example modification code to apply Segment Embedding to [minGPT](https://github.com/karpathy/minGPT).

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, ... )):    
        ...
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.seg_emb = nn.Embedding(2, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))

    ...

    def forward(self, idx, seg, ...:
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        segment_embeddings = self.seg_emb(seg)
        ...
        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + segment_embeddings + position_embeddings)
        ...
```

There's also [Pytorch Lightning version](https://github.com/williamFalcon/minGPT) which fits well with our WaveGAN implementation.

## License

This project is distributed under MIT license.

```
Copyright (c) 2022-present LG AI Research.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## How to cite
```
@misc{kim2022bitters,
  title = {Large-Scale Bidirectional Training for Zero-Shot Image Captioning},
  author = {Kim, Taehoon and Marsden, Mark and Ahn, Pyunghwan and Kim, Sangyun and Lee, Sihaeng and Sala, Alessandra and Kim, Seung Hwan},
  publisher = {arXiv},
  year = {2022},
  doi = {10.48550/ARXIV.2211.06774},
  url = {https://arxiv.org/abs/2211.06774},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}

```

