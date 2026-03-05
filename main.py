from fastapi import FastAPI
from pydantic import BaseModel,  Field
from chain import translator_chain  # <-- use LangChain now
from translator import translate
app = FastAPI()

class TranslateRequest(BaseModel):
    text: str
    num_beams: int = Field(default=1, ge=1, le=8)  # 1 to 8 is a safe beginner range

@app.get("/")
def root():
    return {"message": "Hello from FastAPI"}


@app.post("/translate")
def translate_endpoint(req: TranslateRequest):
    ja = translate(req.text, num_beams=req.num_beams)
    return {"japanese": ja}