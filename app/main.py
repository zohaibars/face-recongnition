
from fastapi import FastAPI
from app.routes import video


app = FastAPI()

app.include_router(video.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="192.168.18.137", port=8008, reload=True)
