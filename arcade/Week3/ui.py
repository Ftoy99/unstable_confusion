import torch
import uvicorn
from fastapi import FastAPI, Form
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from arcade.Week3.TransformerDictionary import TransformerDictionary
from arcade.Week3.translate import translate_functional
from models.transformers.AIAYN import AIAYN, load_model

# Fast api globals
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Torch Device to run model on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformed globals
english_dictionary = TransformerDictionary(name="english")
made_up_dictionary = TransformerDictionary(name="made_up")
model = AIAYN(input_dictionary_size=len(english_dictionary.dictionary) + 1,
              output_dictionary_size=len(made_up_dictionary.dictionary)).to(device)

# Load Model Weights
path_to_weights = "arcade/Week3/weights/AIAYN.pth"
load_model(path_to_weights,model)



@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/translate", response_class=HTMLResponse)
async def translate(text: str = Form(...)):
    sentence = text.split(' ') # Split with spaces

    # Functional translation logic
    functional_translation = " ".join(translate_functional(sentence))

    # Transformer translation logic
    transformer_translation = "".join(["\u2022" + char for char in text])  # Add a dot before each character

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
    uvicorn.run("arcade.Week3.ui:app", host="127.0.0.1", port=9009, reload=True)

if __name__ == '__main__':
    run_ui()