import os
import shutil
import numpy as np
from PIL import Image
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from facenet_pytorch import InceptionResnetV1
import torch
import shutil
from sklearn.cluster import DBSCAN
import warnings
from pathlib import Path

def clustering_design(folder_path = r"detected_faces"):
    warnings.filterwarnings("ignore", category=FutureWarning)

    os.environ["LOKY_MAX_CPU_COUNT"] = "4" 

    # Step 1: Data Preparation
    
    # Create a directory to save face images if it doesn't exist
    if os.path.exists("clusters"):
        shutil.rmtree("clusters")

    Path("clusters").mkdir(parents=True, exist_ok=True)

    # Step 2: Feature Extraction

    # Load pre-trained FaceNet model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Function to extract features from a single image
    def extract_features(image):
        image = image.resize((160, 160))
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        image = image.transpose((0, 3, 1, 2))
        image = torch.tensor(image).float().to(device)
        features = facenet_model(image)
        features = features.detach().cpu().numpy().flatten()
        return features

    # Extract features for each face image in the dataset
    image_paths = []
    face_images = []

    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is an image
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            # Create the image file path
            image_path = os.path.join(folder_path, filename)

            # Open the image using PIL
            image = Image.open(image_path)

            # Convert the image to RGB if it's not already
            image = image.convert("RGB")

            # Add the image and its path to the lists
            face_images.append(image)
            image_paths.append(image_path)

    # Print the image paths
    for path in image_paths:
        print(path)

    # Check if face images are found
    if len(face_images) == 0:
        print("No face images found in the data folder.")
        exit()

    # Step 3: Feature Extraction

    face_features = []
    for i, image in enumerate(face_images):
        # print(f"Extracting features from image {i+1}/{len(face_images)}")
        features = extract_features(image)
        face_features.append(features)

    # Step 4: Normalization

    # Check if face features are found
    if len(face_features) == 0:
        print("No face features extracted.")
        exit()

    # Normalize the extracted features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(face_features)

    # Step 5: Clustering

    # Initialize DBSCAN
    clustering = DBSCAN(eps=0.5, min_samples=5)
    clustering.fit(normalized_features)
    labels = clustering.labels_

    for label in np.unique(labels):
        if label == -1:
            continue
        folder_path = f'clusters/Cluster_{label}'
        os.makedirs(folder_path, exist_ok=True)

    # Move images to respective folders based on cluster assignments
    for i, label in enumerate(labels):
        if label == -1:
            continue
        image_path = image_paths[i]
        cluster_folder = f'clusters/Cluster_{label}'
        shutil.move(image_path, cluster_folder)

    # Fit DBSCAN to normalized features
    #labels = dbscan.fit_predict(normalized_features)

    # # Get number of clusters
    # num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # # Step 6: Folder Organization

    # # Create folders for each cluster
    # for cluster_id in range(num_clusters):
    #     folder_path = f'clusters/Cluster_{cluster_id}'
    #     os.makedirs(folder_path, exist_ok=True)

    # # Move images to respective folders based on cluster assignments
    # for i, label in enumerate(labels):
    #     if label != -1:  # Exclude noise points
    #         image_path = image_paths[i]
    #         cluster_folder = f'clusters/Cluster_{label}'
    #         shutil.move(image_path, cluster_folder)