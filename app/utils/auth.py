from fastapi import HTTPException, Query
from app.utils.settings import API_KEY

def api_key_check(api_key: str = Query(..., title="API Key")):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key