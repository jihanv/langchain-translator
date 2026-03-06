from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Constants
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT_PATH = Path(__file__).parent / "llm_prompt.txt"

# Global Variables
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


def _looks_like_i_am_structure(en: str, ja: str) -> bool:
    """
    Very simple rule for now:
    If English starts with 'I am ...', Japanese should look like '私は ... です/だ/である/という...'

    This is not perfect linguistics—it's a practical filter for obvious paraphrases like:
    "I am a banana." -> "I like bananas."
    """
    en = en.strip().lower()
    if en.startswith("i am "):
        has_subject = ("私は" in ja) or ("わたしは" in ja)
        has_copula = any(x in ja for x in ["です", "だ", "である", "という"])
        return has_subject and has_copula
    return True  # only enforce this rule for "I am ..." sentences for now


def translate_literal_llm(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    _load_once()

    prompt_rules = PROMPT_PATH.read_text(encoding="utf-8").strip()

    # First attempt (normal rules)
    messages = [
        {"role": "system", "content": prompt_rules},
        {"role": "user", "content": f"English: {text}\nJapanese:"},
    ]
    chat_text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(
        chat_text, return_tensors="pt", truncation=True, max_length=512
    ).to(_device)

    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
        )

    generated = out[0][inputs["input_ids"].shape[1]:]
    result = _tokenizer.decode(generated, skip_special_tokens=True).strip()

    # If it violated our simple "I am ..." structure rule, retry once (stricter rules)
    if not _looks_like_i_am_structure(text, result):
        strict_rules = (
            prompt_rules
            + "\nSTRICT: Keep the same meaning and structure. Do not paraphrase.\n"
            + "Output ONLY Japanese. One line only."
        )

        messages = [
            {"role": "system", "content": strict_rules},
            {"role": "user", "content": f"English: {text}\nJapanese:"},
        ]
        chat_text = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _tokenizer(
            chat_text, return_tensors="pt", truncation=True, max_length=512
        ).to(_device)

        with torch.no_grad():
            out = _model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
            )

        generated = out[0][inputs["input_ids"].shape[1]:]
        result = _tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Post-process: keep only the first line (remove explanations)
        result = result.splitlines()[0].strip()

        result = result.strip('「」"')
    return result