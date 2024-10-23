#pytorch
from sqlalchemy import null
import torch
from torchvision import transforms
import time
import os
import subprocess
import pytz

# Set the time zone to Pakistan Standard Time (PKT)
pakistan_timezone = pytz.timezone('Asia/Karachi')

#other lib
# import shutil
import sys
import numpy as np
import base64 
import cv2
import pandas as pd
from datetime import datetime
import json
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#sys.path.insert(0, "scripts/yolov5_face")
sys.path.insert(0, "yolov5_face")
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords
from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, File, Query, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
import os
import shutil
import uvicorn
import concurrent.futures
import asyncio
import datetime



# Check device
app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Variables to store the detected faces and labels for the previous fully processed frame
prev_frame_faces = []
prev_frame_labels = []
person_data = []

images_names = []
images_embs = []
output_dir = "output_videos"
audio_dir = "audio_chunks"
person_data = []
json_data = None
face_detected_data = []
saved_faces = {}
unknown_query_embs = {}
unknown_count = 0

#model = attempt_load("scripts/yolov5_face/yolov5m-face.pt", map_location=device)
model = attempt_load("yolov5_face/yolov5m-face.pt", map_location=device)

# Get model recognition 
from insightface.insight_face import iresnet100
#weight = torch.load("scripts/insightface/resnet100_backbone.pth", map_location = device)
weight = torch.load("insightface/resnet100_backbone.pth", map_location = device)
model_emb = iresnet100()
model_emb.load_state_dict(weight)
model_emb.to(device)
model_emb.eval()

face_preprocess = transforms.Compose([
                                    transforms.ToTensor(), # input PIL => (3,56,56), /255.0
                                    transforms.Resize((112, 112)),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                    ])

# isThread = True
score = 0
name = null


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

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def get_face(input_image):
    # Parameters
    size_convert = 1056
    conf_thres = 0.90
    iou_thres = 0.75

    if input_image is None:
        raise ValueError("Failed to read the input image.")
    # Resize image
    img = resize_image(input_image.copy(), size_convert)

    # Via yolov5-face  
    with torch.no_grad():
        pred = model(img[None, :])[0]

    # Apply NMS
    det = non_max_suppression_face(pred, conf_thres, iou_thres)[0]
    bboxs = np.int32(scale_coords(img.shape[1:], det[:, :4], input_image.shape).round().cpu().numpy())
    landmarks = np.int32(scale_coords_landmarks(img.shape[1:], det[:, 5:15], input_image.shape).round().cpu().numpy())    
    
    return bboxs, landmarks

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

def read_features(root_fearure_path = "static/feature/face_features.npz"):
    data = np.load(root_fearure_path, allow_pickle=True)
    images_name = data["arr1"]
    images_emb = data["arr2"]
    
    return images_name, images_emb

def recognition(face_image, images_names, images_embs):
    global isThread, score, name
    
    # Get feature from face
    query_emb = (get_feature(face_image, training=False))

    scores = (query_emb @ images_embs.T)[0]

    id_min = np.argmax(scores)
    score = scores[id_min]
    name = images_names[id_min]
    return name, score, query_emb

def time_str(total_seconds):
    seconds = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    timestamp_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return timestamp_str

def time_to_seconds(timestamp_str):
    try:
        hours, minutes, seconds = map(int, timestamp_str.split(":"))
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds
    except ValueError:
        raise ValueError("Invalid timestamp format. Use hh:mm:ss")
    
def numpy_array_to_base64(image_array, format='.jpg'):
    _, buffer = cv2.imencode(format, image_array)
    base64_image = base64.b64encode(buffer).decode()
    return base64_image

def extract_audio(video_path, audio_output_path):
    ffmpeg_cmd = [
    'ffmpeg',
    '-y',            
    '-i', video_path,   # Input video file
    '-vn',               # Disable video recording
    '-acodec', 'copy',   # Use the same audio codec as the input
    audio_output_path   # Output audio file
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    
def merge_audio_into_video(video_path, audio_path, output_path):
    ffmpeg_cmd = [
    'ffmpeg',
    '-y',
    '-i', video_path,
    '-i', audio_path,
    '-c:v', 'copy',
    '-map', '0:v:0',
    '-map', '1:a:0',
    '-shortest',
    output_path
    ]
    subprocess.run(ffmpeg_cmd, check=True)

def delete_file(file_path):
    file_path = file_path
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File '{file_path}' has been deleted.")
        else:
            print(f"File '{file_path}' does not exist.")
    except PermissionError as e:
        print(f"PermissionError: {e}")

def delete_mp4_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)
            print(f"Deleted: {file_path}")

def aggregate_entity_info(group):
    # Example: If you want to use the first thumbnail as the aggregated thumbnail
    thumbnail = group['thumbnail'].values[0]

    # Combine and sort timestamps
    timestamps = sorted([ts for sublist in group['timestamps'] for ts in sublist])

    group['startTime'] = pd.to_datetime(group['startTime'])
    group['endTime'] = pd.to_datetime(group['endTime'])

    # Example: If you want to use the minimum start time and maximum end time
    start_time = min(group['startTime'])
    end_time = max(group['endTime'])

    coverage_time = end_time - start_time

    coverage_time = str(coverage_time).split()[-1]
    start_time = str(start_time).split()[-1]
    end_time = str(end_time).split()[-1]


    return pd.Series({
        'thumbnail': thumbnail,
        'name': group['name'].values[0],
        'timestamps': timestamps,
        'coverageTime': coverage_time,
        'startTime': start_time,
        'endTime': end_time,
    })

def remove_unknown_entries():
    data = np.load('static/feature/face_features.npz')

    arr1 = data['arr1']
    arr2 = data['arr2']
    # Create empty lists to store updated data
    updated_arr1 = []
    updated_arr2 = []

    for label, emb in zip(arr1, arr2):
        if not label.startswith('Unknown'):
            updated_arr1.append(label)
            updated_arr2.append(emb)

    # Convert the updated lists to numpy arrays
    updated_arr1 = np.array(updated_arr1)
    updated_arr2 = np.array(updated_arr2)

    # Save the updated data back to the "face_features.npz" file
    np.savez("static/feature/face_features.npz", arr1=updated_arr1, arr2=updated_arr2)

def delete_all_files(directory_path):
    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Check if it's a file (not a subdirectory)
        if os.path.isfile(file_path):
            try:
                # Delete the file
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except OSError as e:
                print(f"Error deleting file: {file_path} - {e}")

def delete_folder(folder_name):
    try:
        shutil.rmtree(folder_name)
        print(f"Folder '{folder_name}' and its contents have been deleted.")
    except FileNotFoundError:
        print(f"Folder '{folder_name}' not found.")
    except PermissionError:
        print(f"Permission denied while trying to delete '{folder_name}'. Make sure you have the necessary permissions.")

def unknown_count_func():  
    if os.path.isfile('unknown_count.txt'):
        with open('unknown_count.txt', 'r') as count_file:
            unknown_count = int(count_file.read())
        return unknown_count
    else:
        unknown_count = 0
        return unknown_count

def update_unknown_count(new_count):
    with open('unknown_count.txt', 'w') as count_file:
        count_file.write(str(new_count))

def update_npz_file(unknown_query_embs):
    data = np.load('static/feature/face_features.npz')
    arr1 = data['arr1']
    arr2 = data['arr2']

    # Create empty lists to store updated data
    updated_arr1 = []
    updated_arr2 = []

    # Add the existing data to the updated lists
    for key, value in unknown_query_embs.items():
        if key not in arr1:
            updated_arr1.extend([key] * len(value))
            updated_arr2.extend(value)

    # Convert the updated lists to numpy arrays
    updated_arr1 = np.array(updated_arr1)
    updated_arr2 = np.array(updated_arr2)

    # Combine the updated data with the existing data
    if arr1 is not None and arr2 is not None:
        updated_arr1 = np.concatenate((arr1, updated_arr1))
        #updated_arr2 = np.concatenate((arr2, updated_arr2))
        if updated_arr2.size == 0:
            updated_arr2 = arr2
        else:
            updated_arr2 = np.vstack((arr2, updated_arr2))

    # Save the updated data back to the "face_features.npz" file
    np.savez("static/feature/face_features.npz", arr1=updated_arr1, arr2=updated_arr2)

def processing_image(input_path ,video_file):
    global json_data
    print("input image",input_path)

    frame = cv2.imread(input_path)

    bboxs, landmarks = get_face(frame)
    results = []
    for i in range(len(bboxs)):
        x1, y1, x2, y2 = bboxs[i]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 146, 230), 2)
        
        face_image = frame[y1:y2, x1:x2]

        images_names, images_embs = read_features()
        
        name, score, query_emb = recognition(face_image, images_names, images_embs)


        if name is None:
                continue
        else:
            if score < 0.35:
                label = "Unknown"
            else:
                label = name.replace("_", " ")    

        thumbnail = numpy_array_to_base64(face_image)

        results.append({'thumbnail': thumbnail, 'Name': label, 'timestamps': '', 'coverageTime' : '', 'startTime': '', 'endTime': ''})
        caption = f"{label}"
        t_size = cv2.getTextSize(caption, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(frame, (x1, y1), (x1 + t_size[0], y1 + t_size[1]), (0, 146, 230), -1)
        cv2.putText(frame, caption, (x1, y1 + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        
    cv2.imwrite(f"output_videos/{video_file.filename}", frame)

    df = pd.DataFrame(results)
    # output_filename = f"output_videos/{video_file.filename}.csv"
    # df.to_csv(output_filename, index=False)

    file_path = "data.json"
    json_data = df.to_json(orient='records')
    
    # Write the data to the JSON file
    with open(file_path, "w") as json_file:
        json.dump(json_data, json_file)
    print("Data saved to:", file_path) 

def processing_chunk(input_path, output_without_audio_path, chunk_timestamp,video_file):

    global person_data,json_data

    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fps = cap.get(cv2.CAP_PROP_FPS)
    size = (frame_width, frame_height)
    video = cv2.VideoWriter(output_without_audio_path, cv2.VideoWriter_fourcc(*'mp4v'), output_fps, size)
    frame_interval = int(output_fps / 3)

    prev_frame_faces, prev_frame_labels = [], []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_seconds = int((position_ms / 1000) + time_to_seconds(chunk_timestamp))
        frame_timestamp = time_str(timestamp_seconds)

        if frame_count % frame_interval != 0 and frame_count != 1:
            for box, label in zip(prev_frame_faces, prev_frame_labels):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 146, 230), 2)
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                cv2.rectangle(frame, (x1, y1), (x1 + t_size[0], y1 + t_size[1]), (0, 146, 230), -1)
                cv2.putText(frame, label, (x1, y1 + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
            video.write(frame)
            continue

        bboxs, landmarks = get_face(frame)
        prev_frame_faces = []
        prev_frame_labels = []
        unknown_persons = []
        
        unknown_count = unknown_count_func()

        for i in range(len(bboxs)):

            x1, y1, x2, y2 = bboxs[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 146, 230), 2)
            
            face_image = frame[y1:y2, x1:x2]

            images_names, images_embs = read_features()
            
            name, score, query_emb = recognition(face_image, images_names, images_embs)

            if name is None:
                continue
            else:
                if score < 0.35:
                    unknown_count += 1
                    label = f"Unknown{unknown_count}"
                    unknown_query_embs[label] = query_emb
                    unknown_persons.append({
                        'query_emb' : query_emb,
                        'label': label,
                        'bbox': bboxs[i],
                        'score': score,
                        'frame_timestamp': frame_timestamp
                    })

                else:
                    label = name.replace("_", " ")

                if label.startswith("Unknown"):
                    # For unknown persons, create separate entries
                    person_entry = {
                        'name': label,
                        'timestamps': [frame_timestamp],  # Initialize with the current timestamp
                        'thumbnail': None,
                        'coverageTime': '00:00:00',
                        'startTime': frame_timestamp,  # Initialize with the current timestamp
                        'endTime': frame_timestamp,
                    }
                    person_data.append(person_entry)
                else:
                    # Find the person in the DataFrame or create a new entry
                    person_entry = next((p for p in person_data if p['name'] == label), None)
                    if person_entry is None:
                        person_entry = {
                            'name': label,
                            'timestamps': [],
                            'thumbnail': None,
                            'coverageTime': '00:00:00',
                            'startTime': None,  # Add 'start time' column
                            'endTime': None,
                        }
                        person_data.append(person_entry)

                # Update the person's data
                person_entry['timestamps'].append(frame_timestamp)
                if person_entry['thumbnail'] is None:
                    person_entry['thumbnail'] = numpy_array_to_base64(face_image)
                if len(person_entry['timestamps']) > 1:
                    if time_to_seconds(person_entry['timestamps'][-1]) - time_to_seconds(person_entry['timestamps'][-2]) <= 5:
                        person_entry['coverageTime'] = time_str(time_to_seconds(person_entry['coverageTime']) + (time_to_seconds(person_entry['timestamps'][-1]) - time_to_seconds(person_entry['timestamps'][-2])))
                
                if person_entry['timestamps']:
                    person_entry['startTime'] = person_entry['timestamps'][0]
                    person_entry['endTime'] = person_entry['timestamps'][-1]

                #caption = f"{label}:{score:.2f}"
                prev_frame_labels.append(label)
                prev_frame_faces.append(bboxs[i])
                caption = f"{label}"
                t_size = cv2.getTextSize(caption, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                cv2.rectangle(frame, (x1, y1), (x1 + t_size[0], y1 + t_size[1]), (0, 146, 230), -1)
                cv2.putText(frame, caption, (x1, y1 + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

        update_npz_file(unknown_query_embs)
        update_unknown_count(unknown_count)    
        video.write(frame)

    video.release()
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)

    # Save the processed video without audio
    print("Video without audio saved at: ", output_without_audio_path)
    df = pd.DataFrame(person_data)
    
    condition = df['timestamps'].apply(len) >= 2
    filtered_df = df[condition]

    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)

    # print("filtered_df columns\n\n",filtered_df.columns)

    filtered_df = filtered_df.groupby('name').apply(aggregate_entity_info).reset_index(drop=True)

    #Load existing 'face_features.npz' file
   
    max_unknown_number = 0
    for name in filtered_df['name']:
        if name.startswith('Unknown'):
            number = int(name[len('Unknown'):])
            max_unknown_number = max(max_unknown_number, number)

    # Save the maximum unknown number to a text file
    with open('unknown_count.txt', 'w') as max_unknown_file:
        max_unknown_file.write(str(max_unknown_number))


    # Convert the DataFrame to a JSON file
    file_path = "data.json"
    json_data = filtered_df.to_json(orient='records')
    
    # Write the data to the JSON file
    with open(file_path, "w") as json_file:
        json.dump(json_data, json_file)
    print("Data saved to:", file_path) 

    # if filtered_df is not None:
    #     output_csv_path = f"output_videos/{video_file.filename}.csv"
    #     #filtered_df_copy = filtered_df.drop('thumbnail', axis=1)
    #     filtered_df.to_csv(output_csv_path, index=False)
    #     print(f"DataFrame saved to '{output_csv_path}'.")
    # else:
    #     print("No person of interest detected!")

def main_func(video_file):
    global images_names, images_embs, output_dir, audio_dir,video_chunk_dir,output_dir

    # Read features
    images_names, images_embs = read_features()
    print("Read features successful")

    # Create a list of timestamps for person data
    label_names = list(set(images_names))
    for n in label_names:
        n = n.replace("_", " ")
        person_entry = {
            'thumbnail': None,
            'name': n,
            'timestamps': [],
            'coverageTime': '00:00:00'
        }
        person_data.append(person_entry)

    output_without_audio_path = video_file.filename
    output_dir = "output_videos"
    audio_dir = "audio_chunks"
    output_chunk_counter = 0

    if not os.path.exists(audio_dir):
        os.mkdir(audio_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    video_chunk_dir = "uploaded_videos"

    # Create the directory if it doesn't exist
    if not os.path.exists(video_chunk_dir):
        os.makedirs(video_chunk_dir)

    uploaded_videos = os.listdir(video_chunk_dir)

    for video_chunk in uploaded_videos:
        if video_chunk.endswith(".mp4"):
            input_path = os.path.join(video_chunk_dir, video_chunk)
            audio_path = os.path.join(audio_dir, f"audio_chunk_{output_chunk_counter}.aac")
            chunk_timestamp = "00:00:00"  # Replace with the actual timestamp

            processing_start = time.time()

            # Extract audio from the video chunk
            extract_audio(input_path, audio_path)

            # Process the video chunk (face recognition and more)
            processing_chunk(input_path, output_without_audio_path, chunk_timestamp, video_file)

            # Merge audio into the output video
            #chunk_output_path = os.path.join(output_dir, f"chunk_output_{output_chunk_counter}.mp4")
            chunk_output_path = os.path.join(output_dir, video_file.filename)
            print("Output path:", chunk_output_path)
            merge_audio_into_video(output_without_audio_path, audio_path, chunk_output_path)
        
            # Clean up temporary files
            delete_file(audio_path)
            delete_file(output_without_audio_path)
            delete_all_files(video_chunk_dir)
            #delete_folder(output_dir)
            

            processing_end = time.time()
            total_processing_time = processing_end - processing_start
            print("Chunk processing time: ", total_processing_time)
            output_chunk_counter += 1
        elif video_chunk.endswith((".mp4", ".png", ".jpg", ".jpeg")):

            input_path = os.path.join(video_chunk_dir, video_chunk)
            processing_image(input_path ,video_file)
            delete_all_files(video_chunk_dir)


#Define a function to validate the API key
async def api_key_check(api_key: str = Query(..., title="API Key")):
    # Replace 'your_api_key' with your actual API key
    if api_key != "1234":
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


# Directory to store uploaded videos and their chunks
UPLOAD_FOLDER = "uploaded_videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

start_main_function = False

# Define a route to upload and split a video
@app.post("/upload_video/")
async def upload_and_split_video(
    api_key: str = Depends(api_key_check),
    video_file: UploadFile = File(...),
):
    print("API key",api_key)
    print("video_file",video_file)
    # Check file extension (you can expand this check)
    if not video_file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.jpg', '.png', '.jpeg')):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    # Save the uploaded video
    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video_file.file, f)

    executor = concurrent.futures.ThreadPoolExecutor()

    await asyncio.get_event_loop().run_in_executor(
        concurrent.futures.ThreadPoolExecutor(),
        main_func,
        video_file,
    )
    current_time = datetime.datetime.now()
    if current_time.time() == datetime.time(0, 0, 0):
        remove_unknown_entries()

    

    if 'json_data' in globals() and json_data:
        return JSONResponse(content=json_data)
    else:
        return JSONResponse(content={"message": "No data available."})

if __name__=="__main__":
    # while True:
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload= True)



