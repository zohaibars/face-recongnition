from decouple import config

API_KEY = config("API_KEY")
MONGO_URI = config("MONGO_URI")
DATABASE_NAME = config("DATABASE_NAME")
COLLECTION_NAME = config("COLLECTION_NAME")
UPLOAD_FOLDER = config("UPLOAD_FOLDER")