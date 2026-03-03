from fastapi import FastAPI
from pydantic import BaseModel
from chain import translator_chain  # <-- use LangChain now

app = FastAPI()

class TranslateRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Hello from FastAPI"}


@app.post("/translate")
def translate_endpoint(req: TranslateRequest):
    ja = translator_chain.invoke(req.text)  # <-- LangChain call
    return {"japanese": ja}