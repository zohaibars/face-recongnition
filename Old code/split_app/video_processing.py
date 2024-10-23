import cv2
import numpy as np
from face_detection import get_face, read_features,recognition, time_str, time_to_seconds, numpy_array_to_base64
import subprocess
import os 
import shutil
import pandas as pd
import time

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
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File '{file_path}' has been deleted.")
    else:
        print(f"File '{file_path}' does not exist.")

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

    group['start time'] = pd.to_datetime(group['start time'])
    group['end time'] = pd.to_datetime(group['end time'])

    # Example: If you want to use the minimum start time and maximum end time
    start_time = min(group['start time'])
    end_time = max(group['end time'])

    coverage_time = end_time - start_time

    coverage_time = str(coverage_time).split()[-1]
    start_time = str(start_time).split()[-1]
    end_time = str(end_time).split()[-1]

    return pd.Series({
        'thumbnail': thumbnail,
        'name': group['name'].values[0],
        'timestamps': timestamps,
        'coverageTime': coverage_time,
        'start time': start_time,
        'end time': end_time,
    })

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

        for i in range(len(bboxs)):

            x1, y1, x2, y2 = bboxs[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 146, 230), 2)

            face_image = frame[y1:y2, x1:x2]
            
            name, score = recognition(face_image, images_names, images_embs)

            if name is None:
                continue
            else:
                if score < 0.35:
                    label = f"Unknown{len(unknown_persons) + 1}"
                    unknown_persons.append({
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
                        'start time': frame_timestamp,  # Initialize with the current timestamp
                        'end time': frame_timestamp
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
                            'start time': None,  # Add 'start time' column
                            'end time': None,
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
                    person_entry['start time'] = person_entry['timestamps'][0]
                    person_entry['end time'] = person_entry['timestamps'][-1]

                caption = f"{label}:{score:.2f}"
                prev_frame_labels.append(label)
                prev_frame_faces.append(bboxs[i])
                t_size = cv2.getTextSize(caption, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                cv2.rectangle(frame, (x1, y1), (x1 + t_size[0], y1 + t_size[1]), (0, 146, 230), -1)
                cv2.putText(frame, caption, (x1, y1 + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

            
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
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    print("person_data \n\n", filtered_df)

    filtered_df = filtered_df.groupby('name').apply(aggregate_entity_info).reset_index(drop=True)

    # Convert the DataFrame to a JSON file
    json_data = filtered_df.to_json(orient='records')

    if filtered_df is not None:
        output_csv_path = f"output_videos/{video_file.filename}.csv"
        #filtered_df_copy = filtered_df.drop('thumbnail', axis=1)
        filtered_df.to_csv(output_csv_path, index=False)
        print(f"DataFrame saved to '{output_csv_path}'.")
    else:
        print("No person of interest detected!")

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

    output_without_audio_path = "output_without_audio.mp4"
    output_dir = "output_videos"
    audio_dir = "audio_chunks"
    output_chunk_counter = 0

    if not os.path.exists(audio_dir):
        os.mkdir(audio_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    video_chunk_dir = "uploaded_videos"
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
            delete_folder(video_chunk_dir)
            #delete_folder(output_dir)
            

            processing_end = time.time()
            total_processing_time = processing_end - processing_start
            print("Chunk processing time: ", total_processing_time)
            output_chunk_counter += 1

