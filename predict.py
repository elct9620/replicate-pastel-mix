# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import configparser
import torch
import pipeline
from cog import BasePredictor, Input, Path
from typing import Iterator

class Predictor(BasePredictor):
    def setup(self):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        config = configparser.ConfigParser()
        config.read("config.ini")

        self.config = pipeline.Config(config)
        self.pipe = pipeline.Pipeline(self.config)

        self.pipe.image2image
        self.pipe.text2image
        self.pipe.upscale

    def __progressive(self, images) -> Iterator[Path]:
        for i, sample in enumerate(images):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            yield Path(output_path)

    def predict(
        self,
        prompt: str = Input(description="Elements to add into image"),
        neg_prompt: str = Input(description="Elements to exclude from image", default=None),
        width: int = Input(description="Width", ge=0, le=1024, default=448),
        height: int = Input(description="Height", ge=0, le=1024, default=640),
        steps: int = Input(description="Steps", ge=0, le=100, default=20),
        guidance: float = Input(description="Higher add more prompt into image", ge=0.0, le=15.0, default=7.0),
        seed: int = Input(description="Seed", default=0),
        hires: bool = Input(description="Generate high resoultion version", default=False)
    ) -> Iterator[Path]:
        seed = torch.seed() if seed == 0 else seed
        generator = torch.Generator('cuda').manual_seed(seed)

        options = {"prompt": prompt,
                   "negative_prompt": neg_prompt,
                   "num_inference_steps": steps,
                   "guidance_scale": guidance,
                   "generator": generator}

        output = self.pipe.text2image(**{
            **options,
            "width": width,
            "height": height
            })

        if any(output.nsfw_content_detected):
            raise Exception(f"NSFW content detected. Try running it again, or try a different prompt.")

        yield from self.__progressive(output.images)

        print(prompt)
        print(f"Negative prompt: {neg_prompt}")
        print(f"Steps: {steps}, Sampler: DPMSolverMultistepScheduler, CFG scale: {guidance}, Size: {width}x{height}")
        print(f"Seed: {generator.initial_seed()}")

        if not hires: return

        upscale = self.pipe.upscale(**{
            **options,
            "image": output.images,
            "num_inference_steps": 20,
            "guidance_scale": 0})

        yield from self.__progressive(upscale.images)

        output = self.pipe.image2image(**{
            **options,
            "image": upscale.images,
            "strength": 0.6})

        yield from self.__progressive(output.images)
