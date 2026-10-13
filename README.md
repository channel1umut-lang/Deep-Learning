# Deep Learning with PyTorch

A progressive series of PyTorch notebooks: from tensor fundamentals and hand-written training loops to CNNs, transfer learning and a **Vision Transformer implemented from scratch** following the original paper.

Every notebook runs in Google Colab (GPU recommended): open it and click **"Open in Colab"** at the top.

**Tech:** Python · PyTorch · torchvision · torchmetrics · torchinfo · TensorBoard · scikit-learn · Matplotlib

---

## ⭐ Highlights

**Vision Transformer (ViT) from scratch**: [`0017_VisionTransformers.ipynb`](0017_VisionTransformers.ipynb)
Patch embedding, multi-head self-attention, MLP and Transformer encoder blocks built as individual `nn.Module`s, assembled into a full ViT using the hyperparameters from *"An Image is Worth 16x16 Words"*. The notebook then compares this with a pretrained **ViT-B/16** fine-tuned on Caltech-101, which reaches **~93% test accuracy**.

**Transfer learning on ImageWoof**: [`0015_TransferLearningImageWoof.ipynb`](0015_TransferLearningImageWoof.ipynb)
Fine-tuning a pretrained **EfficientNet-B0** to classify 10 visually similar dog breeds, with a frozen backbone and a custom classifier head.

**Custom image dataset pipeline**: [`0013_Desert101.ipynb`](0013_Desert101.ipynb)
Loading a folder-based image dataset, transforms and augmentation, and training a custom CNN end to end.

---

## 📚 Notebooks

| # | Topic | What's covered |
|---|---|---|
| 01 | [Data Science Refresher](0001_DataScienceFastReminding.ipynb) | pandas, NumPy and Seaborn recap |
| 02 | [PyTorch Fundamentals](0002_PyTorchFundamentals.ipynb) | Tensors, dtypes, shapes |
| 03 | [Matrix Multiplication](0003_PyTorchMatrixMultiplication.ipynb) | Element-wise vs. matrix ops, shape rules |
| 04 | [Tensor Operations](0004_PyTorchOperations.ipynb) | Arithmetic, broadcasting, NumPy interop |
| 05 | [Aggregation & Manipulation](0005_PyTorchAggregationManipulation.ipynb) | Reshape, stack, squeeze, permute, indexing |
| 06 | [Device Settings](0006_PyTorchDeviceSettings.ipynb) | Device-agnostic code, CPU ↔ GPU |
| 07 | [Training Steps](0007_PyTorchTrainingSteps.ipynb) | Forward pass, loss, backprop, optimizer step |
| 08 | [Structured Training](0008_PyTorchTrainingStructural.ipynb) | Reusable model classes and train/test loops |
| 09 | [Linear vs. Non-Linear](0009_LinearVsNonLinear.ipynb) | Why activation functions matter |
| 10 | [Multi-Class Classification](0010_MultiClassClassification.ipynb) | Softmax, torchmetrics, saving/loading models |
| 11 | [Intro to Computer Vision](0011_IntroToComputerVision.ipynb) | CIFAR-10: linear → non-linear → CNN baselines |
| 12 | [CNN Deep Dive](0012_CNNDeepDive.ipynb) | Conv layers, kernels, strides, padding, pooling |
| 13 | [Desert101: Custom Dataset](0013_Desert101.ipynb) | `ImageFolder`, transforms, custom CNN |
| 14 | [TensorBoard](0014_TensorboardExample.ipynb) | Experiment tracking and logging |
| 15 | [Transfer Learning](0015_TransferLearningImageWoof.ipynb) | EfficientNet-B0 on ImageWoof |
| 17 | [Vision Transformers](0017_VisionTransformers.ipynb) | ViT from scratch + pretrained ViT-B/16 |

---

## ▶️ Running locally

```bash
pip install torch torchvision torchmetrics torchinfo tensorboard matplotlib pandas scikit-learn jupyter
jupyter notebook
```

Some notebooks load data from Google Drive (`/content/drive/...`). To run locally, download the dataset and update the path.
