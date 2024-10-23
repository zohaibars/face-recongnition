import torch
import logging
import numpy as np
from torchvision import transforms
from yolov5_face.models.experimental import attempt_load
from yolov5_face.utils.datasets import letterbox
from yolov5_face.utils.general import check_img_size, non_max_suppression_face, scale_coords
from app.utils.helpers import resize_image, scale_coords_landmarks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load("yolov5_face/yolov5m-face.pt", map_location=device)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_face(input_image):
    size_convert = 1056
    conf_thres = 0.85
    iou_thres = 0.75

    if input_image is None:
        raise ValueError("Failed to read the input image.")

    img = resize_image(input_image.copy(), size_convert, model.stride.max(), device=device)

    logger.info("Getting bounding boxes from model")
    with torch.no_grad():
        pred = model(img[None, :])[0]

    det = non_max_suppression_face(pred, conf_thres, iou_thres)[0]
    bboxs = np.int32(scale_coords(img.shape[1:], det[:, :4], input_image.shape).round().cpu().numpy())
    landmarks = np.int32(scale_coords_landmarks(img.shape[1:], det[:, 5:15], input_image.shape).round().cpu().numpy())    
    
    return bboxs, landmarks
