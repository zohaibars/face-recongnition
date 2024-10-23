import cv2
import os
import json
import logging
import numpy as np
import pandas as pd
from app.services.face_detection import get_face
from app.utils.helpers import numpy_array_to_base64, time_str, time_to_seconds
from app.db.database_fucntions import fetch_all_data, add_name_with_embedding

# Preprocessing and embedding model setup
from torchvision import transforms
import torch
from insightface.insight_face import iresnet100

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight = torch.load("insightface/resnet100_backbone.pth", map_location=device)
model_emb = iresnet100()
model_emb.load_state_dict(weight)
model_emb.to(device)
model_emb.eval()

face_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((112, 112)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def get_feature(face_image, training=True):
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = face_preprocess(face_image).to(device)
    with torch.no_grad():
        emb_img_face = model_emb(face_image[None, :])[0].cpu().numpy() if training else model_emb(face_image[None, :]).cpu().numpy()
    images_emb = emb_img_face / np.linalg.norm(emb_img_face)
    return images_emb

def recognition(face_image, images_names, images_embs):
    logger.info("Performing face recognition!!")
    query_emb = get_feature(face_image, training=False)
    scores = (query_emb @ images_embs.T)[0]
    id_min = np.argmax(scores)
    score = scores[id_min]
    name = images_names[id_min]
    return name, score, query_emb

def processing_image(input_path):
    frame = cv2.imread(input_path)
    bboxs, landmarks = get_face(frame)
    results = []
    images_names, images_embs = fetch_all_data()
    for i in range(len(bboxs)):
        x1, y1, x2, y2 = bboxs[i]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 146, 230), 2)
        face_image = frame[y1:y2, x1:x2]
        name, score, query_emb = recognition(face_image, images_names, images_embs)
        label = "Unknown" if score < 0.35 else name.replace("_", " ")
        thumbnail = numpy_array_to_base64(face_image)
        results.append({'thumbnail': thumbnail, 'Name': label})
    df = pd.DataFrame(results)
    file_path = "data.json"
    json_data = df.to_json(orient='records')
    with open(file_path, "w") as json_file:
        json.dump(json_data, json_file)
    logger.info("Data saved to:", file_path)
    return json_data

def processing_chunk(input_path, chunk_timestamp):
    logger.info("Processing chunks")
    person_data = []
    try:
        cap = cv2.VideoCapture(input_path)
        frame_count = 0
        output_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(output_fps / 3)
        prev_frame_faces, prev_frame_labels = [], []
        unknown_query_embs = {}
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
                continue
            bboxs, landmarks = get_face(frame)
            prev_frame_faces = []
            prev_frame_labels = []
            unknown_persons = []
            unknown_count = get_count()
            images_names, images_embs = fetch_all_data()
            for i in range(len(bboxs)):
                x1, y1, x2, y2 = bboxs[i]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 146, 230), 2)
                face_image = frame[y1:y2, x1:x2]
                name, score, query_emb = recognition(face_image, images_names, images_embs)
                if score < 0.35:
                    unknown_count += 1
                    label = f"Unknown{unknown_count}"
                    unknown_query_embs[label] = query_emb
                    unknown_persons.append({
                        'query_emb': query_emb,
                        'label': label,
                        'bbox': bboxs[i],
                        'score': score,
                        'frame_timestamp': frame_timestamp
                    })
                else:
                    label = name.replace("_", " ")
                if label.startswith("Unknown"):
                    person_entry = {
                        'name': label,
                        'timestamps': [frame_timestamp],
                        'thumbnail': None,
                        'coverageTime': '00:00:00',
                        'startTime': frame_timestamp,
                        'endTime': frame_timestamp,
                    }
                    person_data.append(person_entry)
                else:
                    person_entry = next((p for p in person_data if p['name'] == label), None)
                    if person_entry is None:
                        person_entry = {
                            'name': label,
                            'timestamps': [],
                            'thumbnail': None,
                            'coverageTime': '00:00:00',
                            'startTime': None,
                            'endTime': None,
                        }
                        person_data.append(person_entry)
                person_entry['timestamps'].append(frame_timestamp)
                if person_entry['thumbnail'] is None:
                    person_entry['thumbnail'] = numpy_array_to_base64(face_image)
                if len(person_entry['timestamps']) > 1:
                    if time_to_seconds(person_entry['timestamps'][-1]) - time_to_seconds(person_entry['timestamps'][-2]) <= 5:
                        person_entry['coverageTime'] = time_str(time_to_seconds(person_entry['coverageTime']) + (time_to_seconds(person_entry['timestamps'][-1]) - time_to_seconds(person_entry['timestamps'][-2])))
                if person_entry['timestamps']:
                    person_entry['startTime'] = person_entry['timestamps'][0]
                    person_entry['endTime'] = person_entry['timestamps'][-1]
                prev_frame_labels.append(label)
                prev_frame_faces.append(bboxs[i])
            add_name_with_embedding(unknown_query_embs)
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        logger.error(f"Error encountered during video processing: {e}")

    if person_data:
        df = pd.DataFrame(person_data)
    else:
        json_data = "{}"
        return json_data

    if 'timestamps' in df.columns:
        condition = df['timestamps'].apply(len) >= 2
        filtered_df = df[condition]
        filtered_df = filtered_df.groupby('name').apply(aggregate_entity_info).reset_index(drop=True)
        max_unknown_number = 0
        for name in filtered_df['name']:
            if name.startswith('Unknown'):
                number = int(name[len('Unknown'):])
                max_unknown_number = max(max_unknown_number, number)
        with open('unknown_count.txt', 'w') as max_unknown_file:
            max_unknown_file.write(str(max_unknown_number))
        json_data = filtered_df.to_json(orient='records')
    else:
        json_data = "{}"

    return json_data

def aggregate_entity_info(group):
    thumbnail = group['thumbnail'].values[0]
    timestamps = sorted([ts for sublist in group['timestamps'] for ts in sublist])
    group['startTime'] = pd.to_datetime(group['startTime'])
    group['endTime'] = pd.to_datetime(group['endTime'])
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

def get_count():
    if os.path.isfile('unknown_count.txt'):
        with open('unknown_count.txt', 'r') as count_file:
            unknown_count = int(count_file.read())
        return unknown_count
    else:
        return 0

def main_func(video_file):
    video_chunk_dir = "uploaded_videos"
    images_names, images_embs = fetch_all_data()
    logger.info("Retrieved existing names and embedings successfully")
    person_data = []
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

    if not os.path.exists(video_chunk_dir):
        os.makedirs(video_chunk_dir)

    input_path = os.path.join(video_chunk_dir, video_file.filename)
    chunk_timestamp = "00:00:00"
    json_data = processing_chunk(input_path, chunk_timestamp)
    return json_data