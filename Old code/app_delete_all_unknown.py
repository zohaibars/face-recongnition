from fastapi import FastAPI
from pymongo import MongoClient
from pydantic import BaseModel

app = FastAPI()

client = MongoClient('mongodb://localhost:27017/')
db = client['FacialRecogniton']  # Replace 'your_database_name' with your actual database name
collection = db['FR']

class DeleteResponse(BaseModel):
    deleted_count: int

@app.delete("/delete_unknown", response_model=DeleteResponse)
def delete_all_unknown():
    filter_query = {'name': {'$regex': '^Unknown'}}
    
    # Delete records matching the filter query
    result = collection.delete_many(filter_query)
    
    # Return the number of deleted records
    return DeleteResponse(deleted_count=result.deleted_count)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_delete_all_unknown:app", host="192.168.18.80", port=8009, reload= True)