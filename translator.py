# This file defines a small “translator module”.

# --- 1) Import the tools we need ---
# MarianMTModel = the translation “brain” (a neural network).
# MarianTokenizer = the “text <-> numbers” converter the model needs.
from transformers import MarianMTModel, MarianTokenizer

# torch (PyTorch) is the engine that runs the model’s math on CPU/GPU.
import torch


# --- 2) Choose which pre-trained model to use ---

# This is the name of a model on the Hugging Face Hub (online model library).
# It’s specifically an English -> Japanese translation model.
MODEL_ID = "staka/fugumt-en-ja"


# --- 3) Global variables that we will fill in ONE time (singletons) ---
#   _tokenizer: the tokenizer object
#   _model: the model object
#   _device: where we run the model (Apple GPU or CPU)
_tokenizer = None
_model = None
_device = None


# --- 4) A helper function that loads the model/tokenizer only once ---

def _load_once():
    """
    Load the tokenizer + model a single time.
    This is important for a backend so we don't reload on every request.
    """

    # “global” means: when we assign to these names,
    # we are assigning to the module-level variables above.
    global _tokenizer, _model, _device

    # If we already loaded them before, do nothing.
    # This makes future calls fast.
    if _tokenizer is not None and _model is not None:
        return

    # Download/load the tokenizer for MODEL_ID.
    # The tokenizer is what turns your input text into token IDs (numbers).
    _tokenizer = MarianTokenizer.from_pretrained(MODEL_ID)

    # Download/load the model weights for MODEL_ID.
    # The model is the neural network that will generate a translation.
    _model = MarianMTModel.from_pretrained(MODEL_ID)

    # Decide what device to use:
    # - "mps" = Apple GPU on newer Macs (Metal Performance Shaders)
    # - "cpu" = normal CPU
    _device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Move the model onto that device so it runs there.
    _model.to(_device)


# --- 5) The function you actually call: translate text ---

def translate(text: str) -> str:
    """
    Translate English -> Japanese.
    We keep decoding simple (num_beams=1) to avoid “creative” rewriting.
    """

    # Basic safety check: make sure the input is a string.
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # Make sure the model and tokenizer are loaded.
    # First time: loads them.
    # Later times: returns immediately (fast).
    _load_once()

    # Turn the text into the inputs the model expects.
    # We wrap [text] in a list to make a “batch” of 1 item.
    # return_tensors="pt" means “give me PyTorch tensors”.
    # padding=True makes all items in the batch the same length (useful for batching).
    inputs = _tokenizer(
    [text],
    return_tensors="pt",
    padding=True,
    truncation=True,      # <-- new: cut off input that’s too long
    max_length=512        # <-- new: match model limit
    ).to(_device)

    # We are NOT training the model, just using it.
    # torch.no_grad() tells PyTorch: “don’t track gradients”
    # which saves memory and runs faster.
    with torch.no_grad():

        # Ask the model to generate translated tokens.
        # generate() produces output token IDs (numbers) that represent Japanese text.
        output_ids = _model.generate(
            **inputs,           # same as passing input_ids=..., attention_mask=...
            num_beams=1,        # 1 beam = “greedy” (simple, less fancy)
            do_sample=False,    # no randomness; same input gives same output
            max_new_tokens=120, # cap output length so it can’t go on forever
        )

    # Convert the output token IDs back into readable text.
    # output_ids[0] = the first item in the batch (we only had 1).
    # skip_special_tokens=True removes tokens like <pad> or </s>.
    return _tokenizer.decode(output_ids[0], skip_special_tokens=True)