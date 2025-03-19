# Local Binary Patterns for Classifying Pistachio Species

## 📚 Overview
This project presents an efficient method for classifying pistachio species using Local Binary Patterns (LBP) for feature extraction and machine learning models for classification. The study specifically focuses on two pistachio varieties **Kirmizi**  **Siirt** that have distinct commercial value in the agricultural sector. By leveraging LBP, which captures fine-grained texture features from images, the authors aim to improve accuracy and computational efficiency over more complex deep learning models.

## Methodology

1. Image Preprocessing
- **Dataset**
  - Total images: **2,148**  
  - Kirmizi: **1,232** images  
  - Siirt: **916** images  
  - Split: **70% training / 30% testing** 

2. Feature Extraction
Features extraction can be ferformed using script:
- `LBP_extract_feature.py`: For extracting Local Binary Patterns features.

3. Model Training and Evaluation
- `Training_model.ipynb`:  Using six machine learning models, evaluated on recall, precision, and accuracy metrics.
  - K-Nearest Neighbors (**KNN**)
  - Support Vector Machine (**SVM**)
  - Random Forest (**RF**)
  - Logistic Regression (**LR**)
  - CatBoost (**CB**)
  - Multilayer Perceptron (**MLP**)

## Usage
To use this repository for feature extraction and model training

1. Clone the repository to your local machine.
2. Navigate to the desired script for `feature extraction` and `Traning model`.
4. Run the script to perform feature extraction and train the models.

## Ciation
@inproceedings{bao2024local,
  title={Local Binary Patterns for Classifying Pistachio Species},
  author={Bao Nguyen, Le Huu and Phan, Thi-Thu-Hong},
  booktitle={2024 IEEE International Conference on Consumer Electronics-Asia (ICCE-Asia)},
  pages={1--4},
  year={2024},
  organization={IEEE}
}

You may also access the paper: https://ieeexplore.ieee.org/document/10773854

## Published in: **2024 IEEE International Conference on Consumer Electronics-Asia (ICCE-Asia)**
-**Date Added to IEEE Xplore: 10 December 2024**

-**Conference Location: Danang, Vietnam**

```
Thi-Thu-Hong Phan, Le Huu Bao Nguyen
```