from fastapi import FastAPI

app = FastAPI()


@app.get("/nplr")
async def nplr():
    detections = {}
    return {"detections": detections}

