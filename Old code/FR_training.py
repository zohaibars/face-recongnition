import os
import argparse
import time
#pytorch
from concurrent.futures import thread
from xmlrpc.client import Boolean
from sqlalchemy import null
import torch
from torchvision import transforms
from threading import Thread

#other lib
import sys
import numpy as np
import cv2
import shutil
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, "yolov5_face")
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords
from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
from db_fr import FaceRecDB
db=FaceRecDB()
from fr_mongo_db import *
import uuid
import requests


app = FastAPI()

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get model detect
## Case 1:
# model = attempt_load("scripts/yolov5_face/yolov5s-face.pt", map_location=device)

## Case 2:
model = attempt_load("yolov5_face/yolov5m-face.pt", map_location=device)

# Get model recognition
## Case 1: 
from insightface.insight_face import iresnet100
weight = torch.load("insightface/resnet100_backbone.pth", map_location = device)
model_emb = iresnet100()

## Case 2: 
# from insightface.insight_face import iresnet18
# weight = torch.load("insightface/resnet18_backbone.pth", map_location = device)
# model_emb = iresnet18()

model_emb.load_state_dict(weight)
model_emb.to(device)
model_emb.eval()

face_preprocess = transforms.Compose([
                                    transforms.ToTensor(), # input PIL => (3,56,56), /255.0
                                    transforms.Resize((112, 112)),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                    ])

def resize_image(img0, img_size):
    h0, w0 = img0.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size
    img = letterbox(img0, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    
    return img

def get_face(input_image):
    # Parameters
    size_convert = 256
    conf_thres = 0.8
    iou_thres = 0.8
    
    # Resize image
    img = resize_image(input_image.copy(), size_convert)

    # Via yolov5-face
    with torch.no_grad():
        pred = model(img[None, :])[0]

    # Apply NMS
    det = non_max_suppression_face(pred, conf_thres, iou_thres)[0]
    bboxs = np.int32(scale_coords(img.shape[1:], det[:, :4], input_image.shape).round().cpu().numpy())
    
    return bboxs

def get_feature(face_image, training = True): 
    # Convert to RGB
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    # Preprocessing image BGR
    face_image = face_preprocess(face_image).to(device)
    
    # Via model to get feature
    with torch.no_grad():
        if training:
            emb_img_face = model_emb(face_image[None, :])[0].cpu().numpy()
        else:
            emb_img_face = model_emb(face_image[None, :]).cpu().numpy()
    
    # Convert to array
    images_emb = emb_img_face/np.linalg.norm(emb_img_face)
    return images_emb
import zipfile

def is_zipfile(file_path):
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            return True
    except zipfile.BadZipFile:
        return False

def extract_zip(zip_path, extract_folder):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

def training(zip_file, name_person,is_add_user):

    temp_extract_folder = "temp_extract_folder"
    
    # Ensure the temporary folder exists
    if not temp_extract_folder:
        os.makedirs(temp_extract_folder, exist_ok=True)

    if is_zipfile(zip_file):
        print(f"ZIP file {zip_file} is valid.")
        try:
            extract_zip(zip_file, temp_extract_folder)
            print(f"ZIP file {zip_file} successfully extracted to {temp_extract_folder}.")
        except Exception as e:
            print(f"Error extracting ZIP file {zip_file}: {e}")
    else:
        print(f"Invalid ZIP file: {zip_file}")
        
        #Init results output
    try:
        name_values_dict = {}
        # Read train folder, get and save face
        for root, dirs, files in os.walk(temp_extract_folder):
            # db.add_name(name_person)
            for image in files:
                if image.endswith(("png", "jpg", "jpeg")):
                    input_image = cv2.imread(os.path.join(root, image))  # BGR

                    # Get faces
                    bboxs = get_face(input_image)
                    # Get boxs
                    for i in range(len(bboxs)):
                        # Get location face
                        x1, y1, x2, y2 = bboxs[i]
                        # Get face from location
                        face_image = input_image[y1:y2, x1:x2]
                        # Get feature from face
                        embed=get_feature(face_image, training=True)

                        if name_person not in name_values_dict:
                            name_values_dict[name_person] = []

                        name_values_dict[name_person].append(embed.tolist())
        if is_add_user:
            #add_name_with_embedding(name_values_dict)
            for name, values in name_values_dict.items():
                collection.insert_one({'name': name, 'values': values})
        return True, "Trained Succesully"

    except Exception as e:
        return False,f"Failure {e}"

@app.post("/train")
async def train(training_zip: str = Form(...), name_person: str = Form(...), is_add_user: bool = Form(...)):
    # Save uploaded zip file to a temporary file
    # with open("uploaded_zip.zip", "wb") as zip_file:
    print("Training: ",training_zip,name_person)
    file_path=training_zip
    temp_zip_path = f"temp_zip_{uuid.uuid4()}.zip"
    # file = f"http://192.168.18.80:5006/{response_json}"
    # count=len(os.listdir('videos/geo_news'))+1
    # video_path=video+data['video_name']
    response = requests.get(file_path)
    print("Response is",response)
    # live.save()
    if response.status_code == 200:
        with open(temp_zip_path, 'wb') as f:
            f.write(response.content)
        
    #     extract_to = "static/test_folder"
    # #     zip_file.write(training_zip.file.read())
    
    
    # with open(temp_zip_path, "wb") as temp_zip:
    #     temp_zip.write(training_zip.file.read())

    # print("training \n\n ",training_zip.file, "\n\n")
    # Call your updated training function with the zip file
    result,message=training(temp_zip_path, name_person, is_add_user)
    if result:

        return {"message": "Training Finished Successfully"}
    else:
        return {"message":f"API fails {message}"}

if __name__ == "__main__":
    uvicorn.run("FR_training:app", host="192.168.18.80", port= 8002 , reload = True)