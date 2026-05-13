# EO-SAR Binary Change Detection

Binary pixel-level change detection on paired Electro-Optical (EO) and 
Synthetic Aperture Radar (SAR) imagery using a UNet with ResNet34 backbone.
Built for the GalaxEye Space — Satellite AI Research Intern assessment.

## Approach

- **Architecture**: UNet (segmentation-models-pytorch) with ImageNet-pretrained ResNet34 encoder
- **Input**: 4-channel tensor — RGB pre-event (EO) + grayscale post-event (SAR)
- **Loss**: Combined Dice + BCE with pos_weight=4.3 to handle class imbalance (~19% change pixels)
- **Dataset**: [doron333/change-detection-dataset](https://huggingface.co/datasets/doron333/change-detection-dataset)

## Results

| Metric    | Validation | Test   |
|-----------|-----------|--------|
| IoU       | 0.8545    | 0.4045 |
| Precision | 0.9482    | 0.4415 |
| Recall    | 0.8963    | 0.8284 |
| F1 Score  | 0.9215    | 0.5760 |

## Model Weights

Download from Google Drive: **[best_model.pth](YOUR_GOOGLE_DRIVE_LINK_HERE)**

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (trained on NVIDIA T4)

## Environment Setup

```bash
git clone https://github.com/kiran08012002/galaxeye-change-detection
cd galaxeye-change-detection

conda create -n galaxeye python=3.10 -y
conda activate galaxeye
pip install -r requirements.txt
```

## Dataset Structure

The dataset is loaded automatically from HuggingFace. No manual download needed.
If you prefer local use, place data as:

```
data/
  train/
    images/   # pre-event EO + post-event SAR
    masks/    # binary change masks
  val/
  test/
```

## Training

```bash
python train.py --config config.yaml
```

## Evaluation

```bash
python eval.py --data_path doron333/change-detection-dataset \
               --weights best_model.pth \
               --split test
```

## Citation / References

- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015
- segmentation-models-pytorch: https://github.com/qubvel/segmentation_models.pytorch
- Chen et al., "Remote Sensing Image Change Detection with Transformers", IEEE TGRS 2021
- Daudt et al., "Fully Convolutional Siamese Networks for Change Detection", ICIP 2018
- Dataset: https://huggingface.co/datasets/doron333/change-detection-dataset
