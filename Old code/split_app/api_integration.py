from fastapi import FastAPI, File, Query, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
import os
import shutil
import uvicorn
from video_processing import processing_chunk, delete_file, delete_folder, delete_all_files, merge_audio_into_video
from face_detection import json_data

app = FastAPI()


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
    # Check file extension (you can expand this check)
    if not video_file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    # Save the uploaded video
    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video_file.file, f)


    if 'json_data' in globals() and json_data:
        return JSONResponse(content=json_data)
    else:
        return JSONResponse(content={"message": "No data available."})
    # return JSONResponse(content={"message": "Video uploaded and split successfully"})


# Define a function to run uvicorn in a daemon thread
def run_uvicorn():
    uvicorn.run(app, host="0.0.0.0", port=5000)
