# Masked Autoencoder in PyTorch Lightning
![PyTorch](https://img.shields.io/badge/pytorch-2.0.0-green) [![arXiv](https://img.shields.io/badge/arXiv-2111.06377-b31b1b.svg)](https://arxiv.org/abs/2111.06377)

This repository provides an implementation of the Masked Autoencoder (MAE) framework, a deep learning model for unsupervised representation learning. The MAE model is designed to reconstruct input data from partially masked versions, allowing it to learn meaningful representations that capture important features and patterns in the data. 

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Masked Autocoders](#masked-autoencoders)
- [Training the Masked Autoencoder (MAE) Model](#training-the-masked-autoencoder-mae-model)
- [Converting the MAE Model to a Vision Transformer (ViT)](#converting-the-mae-model-to-a-vision-transformer-vit)


## Overview

The Masked Autoencoders (MAE) training framework uses an approach where random patches of the input image are masked and the missing pixels are reconstructed. This approach is based on two core designs: an asymmetric encoder-decoder architecture and masking a high proportion of the input image.

1. **Asymmetric encoder-decoder architecture:** The encoder operates only on the visible subset of patches (without mask tokens), and a lightweight decoder reconstructs the original image from the latent representation and mask tokens. This design allows for efficient and effective training of large models.

2. **High proportion of input image masking:** Masking around 75% of the input image creates a nontrivial and meaningful self-supervisory task. This encourages learning useful features and reduces redundancy, creating a challenging self-supervisory task that requires a holistic understanding beyond low-level image statistics.

The MAE method has been found to be scalable and effective for visual representation learning, achieving high accuracy and improved generalization performance when used for pre-training data-hungry models like ViT-Large/-Huge on ImageNet-1K. Transfer learning performance in downstream tasks also outperforms supervised pre-training and shows promising scaling behavior.

The original GitHub repository for the project can be found at [this link](https://github.com/facebookresearch/MAE).

The citation for the original paper "Masked Autoencoders Are Scalable Vision Learners" by Kaiming He et al. is as follows:

```bibtex
@article{MaskedAutoencoders2021,
  author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},
  journal = {arXiv:2111.06377},
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  year    = {2021},
}
```

For more in-depth information, please refer to the [original paper](https://arxiv.org/abs/2111.06377) and [GitHub repository](https://github.com/facebookresearch/MAE).

## Installation

To use the MAE framework, follow these steps:

1. Clone this repository to your local machine:

```bash
git clone https://github.com/mtrazzak/lightning-mae.git
```

2. Install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

This command will install the necessary packages, including PyTorch, Lightning, torchvision, timm, and transformers.

## Training the Masked Autoencoder (MAE) Model

To train the MAE model with the default parameters and masking ratio, follow these steps:

1. Prepare your training data: The MAE model expects input data in tensor format. You can preprocess your dataset and convert it into tensors using appropriate techniques and libraries.

2. Import the `MaskedAutoencoderLIT` class:

   ```python
   from mae import MaskedAutoencoder
   ```

3. Initialize the MAE model with the desired parameters. By default, the model uses the 'base' size, 12 input channels, a base learning rate of 3e-5, 4 GPUs, a batch size of 512, 1 warm-up epoch, a weight decay of 0.05, and betas of (0.9, 0.95):

   ```python
   model = MaskedAutoencoder()
   ```

4. Prepare your training data as PyTorch datasets and data loaders. You can utilize `torchvision.datasets` and `torch.utils.data.DataLoader` to load and batch your data:

   ```python
   import torchvision.transforms as transforms
   from torch.utils.data import DataLoader

   transform = transforms.Compose([
       transforms.ToTensor(),
       # Add any additional transformations if needed
   ])

   train_dataset = torchvision.datasets.YourDataset(root='./data', train=True, transform=transform)
   train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
   ```

5. Create a Lightning `Trainer` object and initiate the training process using the `fit` method:

   ```python
   from lightning import Trainer

   trainer = Trainer(max_epochs=10, gpus=1)
   trainer.fit(model, train_dataloader)
   ```

   This will train the MAE model for 10 epochs using the provided training data.

6. After training, save the trained model's state dictionary using `torch.save`:

   ```python
   torch.save(model.state_dict(), 'masked_autoencoder.pt')
   ```

   The trained model can be loaded later for inference or fine-tuning.

## Converting the MAE Model to a Vision Transformer (ViT)

To convert the trained MAE model to a Vision Transformer (ViT) model for fine-tuning, follow these additional steps:

1. Import the `get_vit_from_mae` function:

   ```python
   from mae_to_vit import get_vit_from_mae
   ```

2. Load the trained MAE model using the `MaskedAutoencoder.load_from_checkpoint` method:

   ```python
   trained_model = MaskedAutoencoder.load_from_checkpoint('masked_autoencoder.pt')
   ```

3. Convert the trained MAE model to a ViT model using the `get_vit_from_mae` function:

   ```python
   vit_model = get_vit_from_mae(trained_model.model.state_dict())
   ```

   The `get_vit_from_mae` function takes the pretrained MAE model state dictionary as input and returns a ViT model for fine-tuning. The `vit_model` can be used for further training or downstream tasks.

Please note that you may need to adjust the code and parameters according to your specific use case and dataset.

For more detailed information on the MAE framework, ViT conversion, and customization options, please refer to the code documentation and the source files in this repository.

Happy training!
