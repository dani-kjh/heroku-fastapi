# from io import BytesIO
# import tracemalloc
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from modules.inference import infer_t5

# https://huggingface.co/settings/tokens
# https://huggingface.co/spaces/{username}/{space}/settings
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.head("/")
@app.get("/")
def index() -> FileResponse:
    return FileResponse(path="static/index.html", media_type="text/html")


@app.get("/infer_t5")
def t5(input):
    # tracemalloc.start()
    output = infer_t5(input)
    # print(tracemalloc.get_traced_memory())
    # tracemalloc.stop()
    return {"output": output}

