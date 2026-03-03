from fastapi import FastAPI
from pydantic import BaseModel
from translator import translate

app = FastAPI()

class TranslateRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Hello from FastAPI"}

@app.post("/translate")
def translate_endpoint(req: TranslateRequest):
    ja = translate(req.text)
    return {"japanese": ja}