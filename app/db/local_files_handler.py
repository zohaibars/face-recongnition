import os
import logging
from app.utils.settings import UPLOAD_FOLDER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def delete_uploads():
    logger.info("Deleting excess Files.")
    path = os.getcwd()
    dir_path = os.path.join(path, UPLOAD_FOLDER)
    # Debugging information
    logger.info(f"Directory path: {dir_path}")

    if not os.path.exists(dir_path):
        logger.error(f"Directory does not exist: {dir_path}")
        return
    files = [(os.path.join(dir_path, f), os.path.getmtime(os.path.join(dir_path, f)))
             for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    files.sort(key=lambda x: x[1])

    if len(files) > 50:
        files_to_remove = files[:25]
        for file_path, _ in files_to_remove:
            os.remove(file_path)
        logger.info(f"Deleted 100 files.")
    return "Deleted Successfully!!!"