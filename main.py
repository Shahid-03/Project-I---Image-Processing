import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern

# Parameters for LBP
radius = 1  # Radius for LBP
n_points = 8 * radius  # Number of points for LBP

# Path to your project folder
base_dir = '/Users/shahidibrahim/Desktop/Project-I---Image-Processing'

# Initialize list to store features and labels
features = []
labels = []

# Function to extract LBP features from an image
def extract_lbp_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Apply LBP
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    # Histogram of LBP
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3),
                           range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize
    return hist

# Loop through each emotion category (anger, sadness, etc.)
for emotion_folder in os.listdir(os.path.join(base_dir, 'Train')):
    emotion_path = os.path.join(base_dir, 'Train', emotion_folder)
    
    if os.path.isdir(emotion_path):
        # Loop through each image in the emotion folder
        for image_file in os.listdir(emotion_path):
            image_path = os.path.join(emotion_path, image_file)
            if image_file.endswith('.jpg') or image_file.endswith('.png'):
                # Extract LBP features and append to list
                features.append(extract_lbp_features(image_path))
                labels.append(emotion_folder)

# Convert features and labels to pandas DataFrame
df = pd.DataFrame(features)
df['label'] = labels

# Save to CSV
csv_output_path = os.path.join(base_dir, 'facial_features.csv')
df.to_csv(csv_output_path, index=False)
print(f"Features saved to {csv_output_path}")
