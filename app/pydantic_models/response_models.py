from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    description: str
    
class DeleteResponse(BaseModel):
    deleted_count: int
