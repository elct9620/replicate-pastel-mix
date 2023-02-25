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

        self.__t2i = StableDiffusionPipeline.from_pretrained(self.__config.model_id, cache_dir=self.__config.cache_dir, local_files_only=True,torch_dtype=torch.float16, scheduler=self.__scheduler)
        self.__i2i = StableDiffusionImg2ImgPipeline(**self.__t2i.components)
        self.__upscale = StableDiffusionLatentUpscalePipeline.from_pretrained(self.__config.upscale_model_id, cache_dir=self.__config.cache_dir, local_files_only=True, torch_dtype=torch.float16)

    @property
    @cache
    def text2image(self):
        self.__t2i.enable_attention_slicing()
        self.__t2i.enable_vae_slicing()
        return self.__t2i.to("cuda")

    @property
    @cache
    def image2image(self):
        self.__i2i.enable_xformers_memory_efficient_attention()
        self.__i2i.enable_attention_slicing()
        return self.__i2i.to("cuda")

    @property
    @cache
    def upscale(self):
        self.__upscale.enable_xformers_memory_efficient_attention()
        self.__upscale.enable_attention_slicing()
        return self.__upscale.to("cuda")
