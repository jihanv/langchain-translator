from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel,  Field
from chain import translator_chain  # <-- use LangChain now
from translator import translate as translate_mt
from llm_translator import translate_literal_llm
from typing import Literal
import concurrent.futures
from fastapi import HTTPException

app = FastAPI()

# Healt check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}
class TranslateRequest(BaseModel):
    text: str
    mode: Literal["mt", "llm"] = "mt"
    num_beams: int = Field(default=1, ge=1, le=8)

@app.get("/")
def root():
    return {"message": "Hello from FastAPI"}


@app.post("/translate")
def translate_endpoint(req: TranslateRequest):
    if req.mode == "llm":
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(translate_literal_llm, req.text)
            try:
                ja = future.result(timeout=15)  # seconds
            except concurrent.futures.TimeoutError:
                raise HTTPException(status_code=504, detail="LLM translation timed out")

        return {"japanese": ja, "mode": "llm"}

    ja = translate_mt(req.text, num_beams=req.num_beams)
    return {"japanese": ja, "mode": "mt", "num_beams": req.num_beams}