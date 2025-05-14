---
license: cc-by-nc-sa-4.0
---

## Summary
We introduce the Geospatial Reasoning Segmentation Dataset (GRES), a collection of vision and language data designed around remote-sensing applications. GRES consists of two core components: PreGRES, a dataset consisting of over 1M remote-sensing specific visual instruction-tuning Q/A pairs for pre-training geospatial models, and GRES, a semi-synthetic dataset specialized for reasoning segmentation of remote-sensing data and consisting of 9,205 images and 27,615 natural language queries/answers within those images. From this LISAt dataset, we generate train, test, and validation splits consisting of 7,205, 1,500, and 500 images respectively.

To generate synthetic data, we use the pipeline depicted below. We start with a seed detection dataset (xView). We then filter detections for those that are both visually interesting and highly distinguishable (A). For those detection, we then generate a natural language description (B), and a pixel-wise segmentation mask (C). Finally, the natural language description is used to generate a localization query (D). Together, the segmentation mask and the query form a ground-truth pair for the [LISAT](https://huggingface.co/jquenum/LISAt-7b) reasoning segmentation fine-tuning.

<p align="center">
  <img src="https://github.com/lisat-bair/GRES/blob/main/gres.png" width="1024"/>
  
</p>

## Usage

### 1. Download the [xView 1](https://xviewdataset.org/) dataset.
### 2. Clone this repository.
### 3. Run the command below:

```./extract_gres_images.sh /path/to/xview_train_images /path/to/xView_train.geojson .``` to get the gres image pool.


## LISAT GRES Dataset

This repository contains the LISAT GRES dataset, which includes image files and corresponding annotation files in JSON format. The dataset is organized into three main splits: **train**, **val**, and **test**.

## Dataset Folder Structure

This GRES dataset includes image files and corresponding annotation files in JSON format. The dataset is organized into three main splits: **train**, **val**, and **test**.

```plaintext
├── gres-images/
│   ├── train
│   │   ├── lisat_gres_000000016192.jpg
│   │   ├── lisat_gres_000000016195.jpg
│   │   ├── lisat_gres_000000017340.jpg
│   │   └── ...
│   ├── val
│   │   ├── lisat_gres_000000016203.jpg
│   │   ├── lisat_gres_000000016210.jpg
│   │   ├── lisat_gres_000000017500.jpg
│   │   └── ...
│   ├── test
│   │   ├── lisat_gres_000000016217.jpg
│   │   ├── lisat_gres_000000016234.jpg
│   │   ├── lisat_gres_000000017800.jpg
│   │   └── ...
├── gres-annotations/
│   ├── train
│   │   ├── lisat_gres_000000016192.json
│   │   ├── lisat_gres_000000016195.json
│   │   ├── lisat_gres_000000017340.json
│   │   └── ...
│   │   ├── train.txt
│   ├── val
│   │   ├── lisat_gres_000000016203.json
│   │   ├── lisat_gres_000000016210.json
│   │   ├── lisat_gres_000000017500.json
│   │   └── ...
│   │   ├── val.txt
│   ├── test
│   │   ├── lisat_gres_000000016217.json
│   │   ├── lisat_gres_000000016234.json
│   │   ├── lisat_gres_000000017800.json
│   │   ├── test.txt
│   │   ├── large.txt
│   │   └── small.txt
```


## Citation

If you use LISAt or GRES in your research or applications, please cite our [paper](https://arxiv.org/pdf/2505.02829):

```bibtex
@article{quenum2025lisat,
  title={LISAT: Language-Instructed Segmentation Assistant for Satellite Imagery},
  author={Quenum, Jerome and Hsieh, Wen-Han and Wu, Tsung-Han and Gupta, Ritwik and Darrell, Trevor and Chan, David M},
  journal={arXiv preprint arXiv:2505.02829},
  year={2025}
}
