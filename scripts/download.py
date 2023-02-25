#!/usr/bin/env python

import os
import shutil
import configparser
from diffusers import StableDiffusionPipeline, StableDiffusionLatentUpscalePipeline

config = configparser.ConfigParser()
config.read("config.ini")

MODEL_ID = config.get("generation", "model_id", fallback="andite/pastel-mix")
UPSCALE_MODEL_ID = config.get("upscale", "model_id", fallback="stabilityai/sd-x2-latent-upscaler")
MODEL_CACHE = config.get("cache", "path", fallback="diffusers-cache")

if os.path.exists(MODEL_CACHE):
    shutil.rmtree(MODEL_CACHE)
os.makedirs(MODEL_CACHE, exist_ok=True)

StableDiffusionPipeline.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE)
StableDiffusionLatentUpscalePipeline.from_pretrained(UPSCALE_MODEL_ID, cache_dir=MODEL_CACHE)
