import os
import logging
import shutil
import asyncio
import concurrent.futures
from fastapi.responses import JSONResponse
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Depends,
    Query,
)
from app.utils.settings import UPLOAD_FOLDER
from app.services.video_processing import main_func
from app.utils.auth import api_key_check
from app.pydantic_models.response_models import DeleteResponse
from app.db.database_fucntions import delete_all_unknown
from app.db.local_files_handler import delete_uploads


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/upload_video/")
async def upload_and_split_video(
    api_key: str = Depends(api_key_check),
    video_file: UploadFile = File(...),
):
    delete_uploads()
    if not video_file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.jpg', '.png', '.jpeg')):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
        
    logger.info("File successfully loaded!!")
    
    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video_file.file, f)

    json_data = await asyncio.get_event_loop().run_in_executor(
        concurrent.futures.ThreadPoolExecutor(),
        main_func,
        video_file,
    )
    if json_data:
        return JSONResponse(content=json_data)
    else:
        return JSONResponse(content={"message": "No data available."})
    
    
@router.delete("/delete_unknown", response_model=DeleteResponse)
def delete_unknown():
    deleted_count = delete_all_unknown()
    return DeleteResponse(deleted_count=deleted_count)