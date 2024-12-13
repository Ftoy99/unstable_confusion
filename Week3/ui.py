import os

import torch
from torch import device, cuda
import uvicorn
from fastapi import FastAPI, Form
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from TransformerDictionary import TransformerDictionary
from sequence_helper import pad_sequences
from translate import translate_functional
from AIAYN import AIAYN, load_model

# Fast api globals
app = FastAPI()
# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))  # Absolute path of the script

# Resolve templates path relative to the script directory
templates_dir = os.path.join(script_dir, "templates")

# Check if the path to the templates directory exists
if not os.path.isdir(templates_dir):
    print(f"Warning: templates directory not found at {templates_dir}")

templates = Jinja2Templates(directory=templates_dir)

# Torch Device to run model on
device = device("cuda" if cuda.is_available() else "cpu")

# Transformed globals
english_dictionary = TransformerDictionary(name="english")
made_up_dictionary = TransformerDictionary(name="made_up")
model = AIAYN(input_dictionary_size=len(english_dictionary.dictionary) + 1,
              output_dictionary_size=len(made_up_dictionary.dictionary) + 1).to(device)

# Load Model Weights
path_to_weights = "weights/AIAYN.pth"
load_model(path_to_weights, model)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/translate", response_class=HTMLResponse)
async def translate(text: str = Form(...)):
    sentence = text.split(' ')  # Split with spaces

    # Functional translation logic
    functional_translation_list = translate_functional(sentence)
    functional_translation = " ".join(functional_translation_list)
    # Transformer translation logic

    # Prepare Transformed Inputs in tensors padded etc.
    max_len = len(sentence)
    input_tensor = pad_sequences([[english_dictionary.to_token(x.lower()) for x in sentence]], max_len, 0, device)
    output_tensor = pad_sequences([[]], max_len, 0, device)

    # Translate with transformer
    output_tensor = model(input_tensor, output_tensor)
    output_tensor[:, :, 0] = -float('inf')  # Mask 0 because it has the highest probability
    _, indices = torch.max(output_tensor, dim=-1)
    transformer_translation = indices.squeeze().tolist()
    transformer_translation = [made_up_dictionary.to_word(token) for token in transformer_translation]

    # Return both translations as HTML
    return f"""
    <h2 class='font-semibold'>Functional Translation:</h2>
    <div class='p-4 bg-gray-700 border border-gray-600 rounded text-gray-200'>
        <p>{functional_translation}</p>
    </div>
    
    <h2 class='font-semibold'>Transformer Translation:</h2>
    <div class='p-4 bg-gray-700 border border-gray-600 rounded text-gray-200'>
        <p>{transformer_translation}</p>
    </div>
    """


def run_ui():
    uvicorn.run(app, host="127.0.0.1", port=9009)


if __name__ == '__main__':
    run_ui()
