import base64
import math
from io import BytesIO
from typing import List
from functools import partial

from PIL import Image, ImageDraw
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from imaginairy import LazyLoadingImage, config
from imaginairy.animations import make_bounce_animation
from imaginairy.api import imagine_image_files, imagine, prompt_normalized
from imaginairy.enhancers.describe_image_blip import generate_caption
from imaginairy.enhancers.prompt_expansion import expand_prompts
from imaginairy.log_utils import configure_logging
from imaginairy.prompt_schedules import parse_schedule_strs, prompt_mutator
from imaginairy.schema import ImaginePrompt

configure_logging("ERROR")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def im_2_b64(image):
    buff = BytesIO()
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue())
    return img_str

class ImagineBody(BaseModel):
    prompt_texts: List[str] = ["a photo of a cat"]
    negative_prompt: str = config.DEFAULT_NEGATIVE_PROMPT
    prompt_strength: float = 7.5
    init_image: str = None
    init_image_strength: float = None
    repeats: int = 1
    height: int = 512
    width: int = 512
    steps: int = None
    seed: int = None
    upscale: bool = False
    fix_faces: bool = False
    fix_faces_fidelity: float = 0.5
    sampler_type: str = config.DEFAULT_SAMPLER
    tile: bool = False
    tile_x: bool = False
    tile_y: bool = False
    mask_image: str = None
    mask_prompt: str = None
    mask_mode: str = "replace"
    mask_modify_original: bool = True
    outpaint: str = None
    caption: bool = False
    precision: str = "autocast"
    model_weights_path: str = config.DEFAULT_MODEL
    model_config_path: str = None
    arg_schedules: str = None
class EditBody(ImagineBody):
    model_weights_path: str = "edit"

class DescribeBody(BaseModel):
    images: List[str]




@app.options("/imagine")
def imagine_options():
    # logic for handling OPTIONS requests to /imagine endpoint
    return {"methods": ["POST"], "headers": {"accept": "application/json","Content-Type": "application/json"}}

@app.options("/edit")
def edit_options():
    # logic for handling OPTIONS requests to /edit endpoint
    return {"methods": ["POST"], "headers": {"accept": "application/json","Content-Type": "application/json"}}

@app.options("/describe")
def describe_options():
    # logic for handling OPTIONS requests to /edit endpoint
    return {"methods": ["POST"], "headers": {"accept": "application/json","Content-Type": "application/json"}}

def gen(body):
    #extracted functionality from api.imagine to be used in /imagine and /edit endpoints
    # Instead of saving images to file it adds them to a dict, Removed GIF functionality
    total_image_count = len(body.prompt_texts) * body.repeats
    print(
        f"received {len(body.prompt_texts)} prompt(s) and will repeat them {body.repeats} times to create {total_image_count} images.")

    # load init_image and decode base64 and turn to Pillow Image
    if body.init_image is not None:
        buff = BytesIO(base64.b64decode(body.init_image))
        body.init_image = Image.open(buff)
        print("init", body.init_image.size)

    # load mask_image and decode base64
    if body.mask_image is not None:
        buff = BytesIO(base64.b64decode(body.mask_image))
        body.mask_image = Image.open(buff)
        print("mask", body.mask_image.size)

    if body.init_image_strength is None:
        if body.outpaint or body.mask_image or body.mask_prompt:
            body.init_image_strength = 0
        else:
            body.init_image_strength = 0.6

    prompts = []
    prompt_expanding_iterators = {}

    for _ in range(body.repeats):
        for prompt_text in body.prompt_texts:
            if prompt_text not in prompt_expanding_iterators:
                prompt_expanding_iterators[prompt_text] = expand_prompts(
                    n=math.inf,
                    prompt_text=prompt_text
                )
            prompt_iterator = prompt_expanding_iterators[prompt_text]
            if body.tile:
                _tile_mode = "xy"
            elif body.tile_x:
                _tile_mode = "x"
            elif body.tile_y:
                _tile_mode = "y"
            else:
                _tile_mode = ""

            prompt = ImaginePrompt(
                next(prompt_iterator),
                negative_prompt=body.negative_prompt,
                prompt_strength=body.prompt_strength,
                init_image=body.init_image,
                init_image_strength=body.init_image_strength,
                seed=body.seed,
                sampler_type=body.sampler_type,
                steps=body.steps,
                height=body.height,
                width=body.width,
                mask_image=body.mask_image,
                mask_prompt=body.mask_prompt,
                mask_mode=body.mask_mode,
                mask_modify_original=body.mask_modify_original,
                outpaint=body.outpaint,
                upscale=body.upscale,
                fix_faces=body.fix_faces,
                fix_faces_fidelity=body.fix_faces_fidelity,
                tile_mode=_tile_mode,
                model=body.model_weights_path,
                model_config_path=body.model_config_path,
            )
            if body.arg_schedules:
                schedules = parse_schedule_strs(body.arg_schedules)
                for new_prompt in prompt_mutator(prompt, schedules):
                    prompts.append(new_prompt)
            else:
                prompts.append(prompt)
    results = {}
    for result in imagine(prompts, precision=body.precision,
                          add_caption=body.caption):  # need to add record steps for gif
        prompt = result.prompt
        if prompt.is_intermediate:
            # we don't save intermediate images
            continue
        for image_type in result.images:
            image = result.images[image_type]
            results[image_type] = im_2_b64(image)

    return results

@app.post("/imagine")
def imagine_call(body: ImagineBody):
    print(body)
    return gen(body)

@app.post("/edit")
def edit_options(body: EditBody):
    print("EDIT")
    print(body)
    return gen(body)

@app.post("/describe")
def describe(body: DescribeBody):
    print("DESCRIBE")
    print(body)
    imgs = []
    for b64_str in body.images:
        buff = BytesIO(base64.b64decode(b64_str))
        imgs.append(Image.open(buff))

    descriptions = []
    for img in imgs:
        descriptions.append(generate_caption(img.copy()))
    return {"descriptions": descriptions}


if __name__ == "__main__":
    # start fastAPI app
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
