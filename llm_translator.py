from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

_tokenizer = None
_model = None
_device = None


def _load_once():
    global _tokenizer, _model, _device
    if _tokenizer is not None and _model is not None:
        return

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    _model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    _device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _model.to(_device)


def translate_literal_llm(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    _load_once()

    messages = [
    {
        "role": "system",
        "content": (
            "You are a literal English->Japanese translator for EFL learners.\n"
            "Rules:\n"
            "- Do NOT correct grammar, tense, or word choice.\n"
            "- Preserve grammatical roles exactly.\n"
            "- If English is unnatural, Japanese should also be unnatural in a matching way.\n"
            "- Special rule: If English uses 'I am + -ing adjective' (exciting/interesting/boring), "
            "translate as 'I am a person/thing that makes others feel X' (〜させる存在), NOT as the corrected feeling.\n"
            "- Output ONLY the Japanese translation. No explanations. One line only.\n"
        ),
    },
    # ✅ one example that forces the behavior
    {
        "role": "user",
        "content": "Translate literally.\nEnglish: I am exciting.\nJapanese:",
    },
    {
        "role": "assistant",
        "content": "私は人を興奮させる存在です。",
    },
    # ✅ now the real request
    {
        "role": "user",
        "content": f"Translate literally.\nEnglish: {text}\nJapanese:",
    },
    ]
    chat_text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(chat_text, return_tensors="pt", truncation=True, max_length=512).to(_device)

    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
        )

    generated = out[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(generated, skip_special_tokens=True).strip()