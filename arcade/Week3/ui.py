import uvicorn
from fastapi import FastAPI, Form
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/translate", response_class=HTMLResponse)
async def translate(text: str = Form(...)):
    # Functional translation logic (simple example)
    functional_translation = text[::-1]  # Reverse the text

    # Transformer translation logic (placeholder example)
    transformer_translation = "".join(["\u2022" + char for char in text])  # Add a dot before each character

    # Return both translations as HTML
    return f"""
    <div class='p-4 bg-gray-700 border border-gray-600 rounded text-gray-200'>
        <h2 class='font-semibold'>Functional Translation:</h2>
        <p>{functional_translation}</p>
    </div>
    <div class='p-4 bg-gray-700 border border-gray-600 rounded text-gray-200'>
        <h2 class='font-semibold'>Transformer Translation:</h2>
        <p>{transformer_translation}</p>
    </div>
    """

if __name__ == '__main__':
    uvicorn.run("arcade.Week3.ui:app", host="127.0.0.1", port=9009, reload=True)