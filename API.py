import base64
import math
import zipfile
from io import BytesIO
from typing import List, Optional
from functools import partial

from PIL import Image, ImageDraw
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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

description = """
Imaginairy API allows you to access the functionality of imaiginary from a web browser or other client.

## Usage
There are three endpoints:
- `/imagine` -> generates an image/ same functionality as the imagine command from the original repo
- `/edit` -> edits an image using instructPix2Pix/ same functionality as the edit command from the original repo
- `/describe` -> uses the CLIP model to describe an image/ same functionality as the describe command from the original repo

### Notes
- When using multiple prompts, the docs page automatically concatenates them creating unwanted behaviour. To avoid this, use the `curl` command or another client and add a seperate form for each prompt, e.g.`-F 'prompt_texts=prompt1' -F 'prompt_texts=prompt2'`. 
- Both the **/imagine** and **/edit** endpoints support responding with a single image or multiple images. When returning one image, it is sent as a byte stream, when returning multiple images, it is sent as a zip file.
"""

app = FastAPI(title="ImaginAIry API", description=description)
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


class DescribeBody(BaseModel):
    images: List[str]


@app.options("/imagine")
def imagine_options():
    # logic for handling OPTIONS requests to /imagine endpoint
    return {"methods": ["POST"], "headers": {"accept": "application/json", "Content-Type": "application/json"}}


@app.options("/edit")
def edit_options():
    # logic for handling OPTIONS requests to /edit endpoint
    return {"methods": ["POST"], "headers": {"accept": "application/json", "Content-Type": "application/json"}}


@app.options("/describe")
def describe_options():
    # logic for handling OPTIONS requests to /edit endpoint
    return {"methods": ["POST"], "headers": {"accept": "application/json", "Content-Type": "application/json"}}


def gen(prompt_texts: List[str] = ["a photo of a cat"],
        negative_prompt: str = config.DEFAULT_NEGATIVE_PROMPT,
        prompt_strength: float = 7.5,
        init_image: Optional[UploadFile] = None,
        init_image_b64: Optional[str] = None,
        init_image_strength: Optional[float] = None,
        repeats: int = 1,
        height: int = 512,
        width: int = 512,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        upscale: bool = False,
        fix_faces: bool = False,
        fix_faces_fidelity: float = 0.5,
        sampler_type: str = config.DEFAULT_SAMPLER,
        tile: bool = False,
        tile_x: bool = False,
        tile_y: bool = False,
        mask_image: Optional[UploadFile] = None,
        mask_image_b64: Optional[str] = None,
        mask_prompt: Optional[str] = None,
        mask_mode: str = "replace",
        mask_modify_original: bool = True,
        outpaint: Optional[str] = None,
        caption: bool = False,
        precision: str = "autocast",
        model_weights_path: str = config.DEFAULT_MODEL,
        model_config_path: Optional[str] = None,
        arg_schedules: Optional[str] = None,
        collect_results=False,
        image_key="generated"):
    # extracted functionality from api.imagine to be used in /imagine and /edit endpoints
    # Instead of saving images to file it adds them to a dict, Removed GIF functionality
    total_image_count = len(prompt_texts) * repeats
    print(
        f"received {len(prompt_texts)} prompt(s) and will repeat them {repeats} times to create {total_image_count} images.")

    if init_image is not None and init_image_b64 is not None:
        return {"error": "init_image and init_image_b64 cannot both be set"}

    if mask_image is not None and mask_image_b64 is not None:
        return {"error": "mask_image and mask_image_b64 cannot both be set"}

    # load init_image and decode base64 and turn to Pillow Image
    if init_image_b64 is not None:
        buff = BytesIO(base64.b64decode(init_image_b64))
        init_image = Image.open(buff)
        print("init", init_image.size)

    # load mask_image and decode base64
    if mask_image_b64 is not None:
        buff = BytesIO(base64.b64decode(mask_image_b64))
        mask_image = Image.open(buff)
        print("mask", mask_image.size)

    if init_image_strength is None:
        if outpaint or mask_image or mask_prompt:
            init_image_strength = 0
        else:
            init_image_strength = 0.6

    prompts = []
    prompt_expanding_iterators = {}

    for _ in range(repeats):
        for prompt_text in prompt_texts:
            if prompt_text not in prompt_expanding_iterators:
                prompt_expanding_iterators[prompt_text] = expand_prompts(
                    n=math.inf,
                    prompt_text=prompt_text
                )
            prompt_iterator = prompt_expanding_iterators[prompt_text]
            if tile:
                _tile_mode = "xy"
            elif tile_x:
                _tile_mode = "x"
            elif tile_y:
                _tile_mode = "y"
            else:
                _tile_mode = ""

            prompt = ImaginePrompt(
                next(prompt_iterator),
                negative_prompt=negative_prompt,
                prompt_strength=prompt_strength,
                init_image=init_image,
                init_image_strength=init_image_strength,
                seed=seed,
                sampler_type=sampler_type,
                steps=steps,
                height=height,
                width=width,
                mask_image=mask_image,
                mask_prompt=mask_prompt,
                mask_mode=mask_mode,
                mask_modify_original=mask_modify_original,
                outpaint=outpaint,
                upscale=upscale,
                fix_faces=fix_faces,
                fix_faces_fidelity=fix_faces_fidelity,
                tile_mode=_tile_mode,
                model=model_weights_path,
                model_config_path=model_config_path,
            )
            if arg_schedules:
                schedules = parse_schedule_strs(arg_schedules)
                for new_prompt in prompt_mutator(prompt, schedules):
                    prompts.append(new_prompt)
            else:
                prompts.append(prompt)
    if collect_results:
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w") as archive:

            for i, result in enumerate(imagine(prompts, precision=precision,
                                               add_caption=caption)):  # need to add record steps for gif
                prompt = result.prompt
                if prompt.is_intermediate:
                    # we don't save intermediate images
                    continue
                # for image_type in result.images:
                for image_key in result.images:
                    image = result.images[image_key]
                    image_buffer = BytesIO()
                    image.save(image_buffer, format="JPEG")
                    archive.writestr(f"{image_key}{i + 1}.jpg", image_buffer.getvalue())
        zip_buffer.seek(0)
        return StreamingResponse(zip_buffer, media_type="application/zip")

    else:
        if len(prompts) > 1:
            # Zip results and return as StreamingResponse
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, mode="w") as archive:

                for i, result in enumerate(imagine(prompts, precision=precision,
                                                   add_caption=caption)):  # need to add record steps for gif
                    prompt = result.prompt
                    if prompt.is_intermediate:
                        # we don't save intermediate images
                        continue
                    # for image_type in result.images:
                    image = result.images[image_key]
                    image_buffer = BytesIO()
                    image.save(image_buffer, format="JPEG")
                    archive.writestr(f"{image_key}{i + 1}.jpg", image_buffer.getvalue())
            zip_buffer.seek(0)
            return StreamingResponse(zip_buffer, media_type="application/zip")
        else:
            # Return single image
            prompt = prompts[0]
            result = next(imagine([prompt], precision=precision, add_caption=caption))
            image = result.images[image_key]
            image_buffer = BytesIO()
            image.save(image_buffer, format="JPEG")
            image_buffer.seek(0)
            return StreamingResponse(image_buffer, media_type="image/jpeg")


@app.post("/imagine")
def imagine_call(prompt_texts: List[str] = Form(...,
                                                description="List of prompts to use for editing, when cURL is used, add a new form for every prompt i.e. `-F 'prompt_texts=prompt1' -F 'prompt_texts=prompt2'`"),
                 negative_prompt: str = Form(config.DEFAULT_NEGATIVE_PROMPT,
                                             description="Negative prompt, making things less likely to appear in the image"),
                 prompt_strength: float = Form(7.5,
                                               description="Strength of the prompt, higher values make the prompt have more effect, default is 7.5"),
                 init_image: Optional[UploadFile] = None,
                 init_image_b64: Optional[str] = Form(None, description="Base64 encoded image to use as initial image"),
                 init_image_strength: Optional[float] = Form(None,
                                                             description="Strength of the initial image, higher values make the initial image have more effect, default is 0.5"),
                 repeats: int = Form(1, description="Number of times to repeat the prompt, default is 1"),
                 height: int = Form(512, description="Height of the image, default is 512"),
                 width: int = Form(512, description="Width of the image, default is 512"),
                 steps: Optional[int] = Form(None, description="Number of diffusion steps"),
                 seed: Optional[int] = Form(None, description="Seed for the random number generator"),
                 upscale: bool = Form(False, description="Upscale the image by 2x"),
                 fix_faces: bool = Form(False, description="Fix faces in the image"),
                 fix_faces_fidelity: float = Form(0.5,
                                                  description="Fidelity of the face fixer, higher values make the face fixer more accurate, default is 0.5"),
                 sampler_type: str = Form(config.DEFAULT_SAMPLER,
                                          description="Sampler to use for the image generation, default is `k_dpmpp_2m`, other options are: [`plms, ddim, k_dpm_fast, k_dpm_adaptive, k_lms, k_dpm_2, k_dpm_2_a, k_dpmpp_2m, k_dpmpp_2s_a, k_euler, k_euler_a, k_heun`]"),
                 tile: bool = Form(False, description="Make image tilable in both x and y directions"),
                 tile_x: bool = Form(False, description="Make image tilable in x direction"),
                 tile_y: bool = Form(False, description="Make image tilable in y direction"),
                 mask_image: Optional[UploadFile] = None,
                 mask_image_b64: Optional[str] = Form(None, description="Base64 encoded image to use as mask image"),
                 mask_prompt: Optional[str] = Form(None, description="Prompt to use to define what should be masked"),
                 mask_mode: str = Form("replace",
                                       description="Mode to use for the mask, default is `replace`, other options is `keep`"),
                 outpaint: Optional[str] = Form(None,
                                                description="Specify in what directions to expand the image. Values will be snapped such that output image size is multiples of 8. Example: up10,down300,left50,right50/ u10,d300,l50,r50/ all200/ a200  "),
                 caption: bool = Form(False, description="Add caption to the image, added to the bottom of the image"),
                 precision: str = Form("autocast",
                                       description="Precision to use for the image generation, default is `autocast`, other option is `full`]"),
                 model_weights_path: str = Form(config.DEFAULT_MODEL,
                                                description="Model weights to use, default is `SD-1.5`, options available out of the box: `SD-1.4, SD-1.5-inpaint, SD-2.0, SD-2.0-inpaint, SD-2.0-v, SD-2.0-depth, SD-2.1, SD-2.1-inpaint, SD-2.1-v, openjourney-v1, openjourney-v2`"),
                 model_config_path: Optional[str] = Form(None,
                                                         description="Model config file to use. If a model name is specified, the appropriate config will be used. (only necessary when using a custom model)"),
                 arg_schedules: Optional[str] = Form(None,
                                                     description="Schedule how an argument should change over several generations. Format: `arg_name[start:end:increment]` or `arg_name[val,val2,val3]`"),
                 collect_results: bool = Form(False,
                                              description="Based on the query parameters `imaginary` stores multiple images in the result, normally only the final result is send in the response, using this all internal images (if any) are stored in the result and send as a zip file"),
                 ):
    # if init_image is not none load image an read to PIL file
    if init_image is not None:
        buff = BytesIO()
        buff.write(init_image.file.read())
        buff.seek(0)
        init_image = Image.open(buff)

    if mask_image is not None:
        buff = BytesIO()
        buff.write(mask_image.file.read())
        buff.seek(0)
        mask_image = Image.open(buff)

    image_key = "generated"
    if upscale:
        image_key = "upscaled"

    return gen(
        prompt_texts=prompt_texts,
        negative_prompt=negative_prompt,
        prompt_strength=prompt_strength,
        init_image=init_image,
        init_image_b64=init_image_b64,
        init_image_strength=init_image_strength,
        repeats=repeats,
        height=height,
        width=width,
        steps=steps,
        seed=seed,
        upscale=upscale,
        fix_faces=fix_faces,
        fix_faces_fidelity=fix_faces_fidelity,
        sampler_type=sampler_type,
        tile=tile,
        tile_x=tile_x,
        tile_y=tile_y,
        mask_image=mask_image,
        mask_image_b64=mask_image_b64,
        mask_prompt=mask_prompt,
        mask_mode=mask_mode,
        outpaint=outpaint,
        caption=caption,
        precision=precision,
        model_weights_path=model_weights_path,
        model_config_path=model_config_path,
        arg_schedules=arg_schedules,
        collect_results=collect_results,
        image_key=image_key,
    )


@app.post("/edit")
def edit_options(prompt_texts: List[str] = Form(...,
                                                description="List of prompts to use for editing, when cURL is used, add new form for every prompt i.e. `-F 'prompt_texts=prompt1' -F 'prompt_texts=prompt2'`"),
                 negative_prompt: str = Form(config.DEFAULT_NEGATIVE_PROMPT,
                                             description="Negative prompt, making things less likely to appear in the image"),
                 prompt_strength: float = Form(7.5,
                                               description="Strength of the prompt, higher values make the prompt have more effect, default is 7.5"),
                 init_image: Optional[UploadFile] = File(..., description="Image to use as initial image"),
                 init_image_b64: Optional[str] = Form(None, description="Base64 encoded image to use as initial image"),
                 init_image_strength: Optional[float] = Form(None,
                                                             description="Strength of the initial image, higher values make the initial image have more effect, default is 0.5"),
                 repeats: int = Form(1, description="Number of times to repeat the prompt, default is 1"),
                 height: int = Form(512, description="Height of the image, default is 512"),
                 width: int = Form(512, description="Width of the image, default is 512"),
                 steps: Optional[int] = Form(None, description="Number of diffusion steps"),
                 seed: Optional[int] = Form(None, description="Seed for the random number generator"),
                 upscale: bool = Form(False, description="Upscale the image by 2x"),
                 fix_faces: bool = Form(False, description="Fix faces in the image"),
                 fix_faces_fidelity: float = Form(0.5,
                                                  description="Fidelity of the face fixer, higher values make the face fixer more accurate, default is 0.5"),
                 sampler_type: str = Form(config.DEFAULT_SAMPLER,
                                          description="Sampler to use for the image generation, default is `k_dpmpp_2m`, other options are: [`plms, ddim, k_dpm_fast, k_dpm_adaptive, k_lms, k_dpm_2, k_dpm_2_a, k_dpmpp_2m, k_dpmpp_2s_a, k_euler, k_euler_a, k_heun`]"),
                 tile: bool = Form(False, description="Make image tilable in both x and y directions"),
                 tile_x: bool = Form(False, description="Make image tilable in x direction"),
                 tile_y: bool = Form(False, description="Make image tilable in y direction"),
                 mask_image: Optional[UploadFile] = None,
                 mask_image_b64: Optional[str] = Form(None, description="Base64 encoded image to use as mask image"),
                 mask_prompt: Optional[str] = Form(None, description="Prompt to use to define what should be masked"),
                 mask_mode: str = Form("replace",
                                       description="Mode to use for the mask, default is `replace`, other options is `keep`"),
                 outpaint: Optional[str] = Form(None,
                                                description="Specify in what directions to expand the image. Values will be snapped such that output image size is multiples of 8. Example: up10,down300,left50,right50/ u10,d300,l50,r50/ all200/ a200  "),
                 caption: bool = Form(False, description="Add caption to the image, added to the bottom of the image"),
                 precision: str = Form("autocast",
                                       description="Precision to use for the image generation, default is `autocast`, other option is `full`]"),
                 model_config_path: Optional[str] = Form(None,
                                                         description="Model config file to use. If a model name is specified, the appropriate config will be used. (only necessary when using a custom model)"),
                 arg_schedules: Optional[str] = Form(None,
                                                     description="Schedule how an argument should change over several generations. Format: `arg_name[start:end:increment]` or `arg_name[val,val2,val3]`"),
                 collect_results: bool = Form(False,
                                              description="Based on the query parameters `imaginary` stores multiple images in the result, normally only the final result is send in the response, using this all internal images (if any) are stored in the result and send as a zip file"),
                 ):
    if init_image is not None:
        buff = BytesIO()
        buff.write(init_image.file.read())
        buff.seek(0)
        init_image = Image.open(buff)

    if mask_image is not None:
        buff = BytesIO()
        buff.write(mask_image.file.read())
        buff.seek(0)
        mask_image = Image.open(buff)

    image_key = "generated"
    if upscale:
        image_key = "upscaled"

    return gen(
        prompt_texts=prompt_texts,
        negative_prompt=negative_prompt,
        prompt_strength=prompt_strength,
        init_image=init_image,
        init_image_b64=init_image_b64,
        init_image_strength=init_image_strength,
        repeats=repeats,
        height=height,
        width=width,
        steps=steps,
        seed=seed,
        upscale=upscale,
        fix_faces=fix_faces,
        fix_faces_fidelity=fix_faces_fidelity,
        sampler_type=sampler_type,
        tile=tile,
        tile_x=tile_x,
        tile_y=tile_y,
        mask_image=mask_image,
        mask_image_b64=mask_image_b64,
        mask_prompt=mask_prompt,
        mask_mode=mask_mode,
        outpaint=outpaint,
        caption=caption,
        precision=precision,
        model_weights_path="edit",
        model_config_path=model_config_path,
        arg_schedules=arg_schedules,
        collect_results=collect_results,
        image_key=image_key
    )


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
    # data = ImagineBody(prompt_texts = ["a green lush forrest"])
    # response = imagine_call(data)
    # print(response)
