from face_detection import read_features, images_names, images_embs
from video_processing import main_func
from api_integration import upload_and_split_video

if __name__ == "__main__":
    main_func(upload_and_split_video.video_file.filename)