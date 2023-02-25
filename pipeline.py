import torch
from functools import cache
from configparser import ConfigParser
from diffusers import StableDiffusionPipeline, StableDiffusionLatentUpscalePipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

class Config:
    def __init__(self, config: ConfigParser):
        self.__config = config

    @property
    def model_id(self):
        return self.__config.get("generation", "model_id", fallback="andite/pastel-mix")

    @property
    def upscale_model_id(self):
        return self.__config.get("upscale", "model_id", fallback="stabilityai/sd-x2-latent-upscaler")

    @property
    def cache_dir(self):
        return self.__config.get("cache", "path", fallback="diffusers-cache")

class Pipeline:
    def __init__(self, config: Config):
        self.__config = config
        self.__scheduler = DPMSolverMultistepScheduler.from_pretrained(config.model_id, subfolder="scheduler", cache_dir=config.cache_dir, local_files_only=True)

    @property
    @cache
    def text2image(self):
        pipe = StableDiffusionPipeline.from_pretrained(self.__config.model_id, cache_dir=self.__config.cache_dir, local_files_only=True,torch_dtype=torch.float16, scheduler=self.__scheduler).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        return pipe

    @property
    @cache
    def image2image(self):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.__config.model_id, cache_dir=self.__config.cache_dir, local_files_only=True,torch_dtype=torch.float16, scheduler=self.__scheduler).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_attention_slicing()
        return pipe

    @property
    @cache
    def upscale(self):
        pipe = StableDiffusionLatentUpscalePipeline.from_pretrained(self.__config.upscale_model_id, cache_dir=self.__config.cache_dir, local_files_only=True, torch_dtype=torch.float16).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_attention_slicing()
        return pipe
