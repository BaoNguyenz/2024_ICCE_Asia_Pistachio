import os
import cv2
import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io
from skimage import feature
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import skew, kurtosis, entropy

# set PATH=C:\Users\baohu\AppData\Local\Programs\Python\Python310;%PATH%

# python --version

# Python 3.10.0

# python your_file.py


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def read_all_image_paths(parent_folder):
    all_image_paths = []
    
    for class_folder in sorted(os.listdir(parent_folder), key=natural_sort_key):
        class_folder_path = os.path.join(parent_folder, class_folder)
        if os.path.isdir(class_folder_path):
            image_names = sorted(os.listdir(class_folder_path), key=natural_sort_key)
            
            if not image_names:
                print(f"No images found in {class_folder_path}.")
                continue
            
            print(f"Found {len(image_names)} images in '{class_folder}'. Showing first 3 images:")
            print(image_names[:3])
            
            image_paths = [os.path.join(class_folder_path, name) for name in image_names]
            all_image_paths.extend(image_paths)
    
    print(f"\nTotal images collected: {len(all_image_paths)}\n")
    return all_image_paths

def create_output_folder(folder_name="save_output"):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"📁 Created output folder: {folder_name}")
    else:
        print(f"📂 Output folder already exists: {folder_name}")
    return folder_name

# COMPUTE LBP FEATURES
def compute_lbp_features(image_paths, points=8, radius=1, output_folder="save_output"):
    output_dir = create_output_folder(output_folder)

    features_csv = os.path.join(output_dir, "lbp_features.csv")
    labels_csv = os.path.join(output_dir, "lbp_labels.csv")

    lbp_data = []
    label_data = []

    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is not None:
            lbp = feature.local_binary_pattern(image, points, radius, method="uniform")

            (hist, _) = np.histogram(
                lbp.ravel(),
                bins=np.arange(0, points + 3),
                range=(0, points + 2)
            )

            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)

            image_name = os.path.basename(image_path)
            class_name = os.path.basename(os.path.dirname(image_path))

            feature_row = [image_name] + hist.tolist()
            label_row = [image_name, class_name]

            lbp_data.append(feature_row)
            label_data.append(label_row)

        else:
            print(f"❌ Failed to read {image_path}")
            image_name = os.path.basename(image_path)
            class_name = os.path.basename(os.path.dirname(image_path))

            feature_row = [image_name] + [np.nan] * (points + 2)
            label_row = [image_name, class_name]

            lbp_data.append(feature_row)
            label_data.append(label_row)

    feature_columns = ["Name"] + [f"lbp_{i}" for i in range(points + 2)]
    label_columns = ["Name", "class"]

    lbp_features_df = pd.DataFrame(lbp_data, columns=feature_columns)
    lbp_labels_df = pd.DataFrame(label_data, columns=label_columns)

    lbp_features_df.to_csv(features_csv, index=False)
    lbp_labels_df.to_csv(labels_csv, index=False)

    print(f"\n✅ LBP features saved to: {features_csv}")
    print(f"✅ LBP labels saved to: {labels_csv}")

    return lbp_features_df, lbp_labels_df


dataset_folder = r'D:\PROJECTWORSHOP\Pistachio\2024_ICCE_Asia_Pistachio\Datasets'

# Take all path
image_paths = read_all_image_paths(dataset_folder)

# ===== LBP =====
lbp_features_df, lbp_labels_df = compute_lbp_features(
    image_paths,
    points=8, 
    radius=1,
    output_folder="save_output"
)





