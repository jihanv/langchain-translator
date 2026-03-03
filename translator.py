# translator.py
from transformers import MarianMTModel, MarianTokenizer
import torch

MODEL_ID = "Helsinki-NLP/opus-mt-en-jap"

# These are module-level “singletons” (loaded once)
_tokenizer = None
_model = None
_device = None


def _load_once():
    """
    Load the tokenizer + model a single time.
    This is important for a backend so we don't reload on every request.
    """
    global _tokenizer, _model, _device

    if _tokenizer is not None and _model is not None:
        return

    _tokenizer = MarianTokenizer.from_pretrained(MODEL_ID)
    _model = MarianMTModel.from_pretrained(MODEL_ID)

    # Use Apple GPU (MPS) if available, otherwise CPU
    _device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _model.to(_device)


def translate(text: str) -> str:
    """
    Translate English -> Japanese.
    We keep decoding simple (num_beams=1) to avoid “creative” rewriting.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    _load_once()

    inputs = _tokenizer([text], return_tensors="pt", padding=True).to(_device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            num_beams=1,        # simpler / less “polished” than beam search
            do_sample=False,
            max_new_tokens=120,
        )

    return _tokenizer.decode(output_ids[0], skip_special_tokens=True)