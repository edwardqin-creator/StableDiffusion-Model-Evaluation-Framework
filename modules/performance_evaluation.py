import sys
sys.path.append('../')
import os
import time
import requests
from zipfile import ZipFile
from PIL import Image
import numpy as np

import torch
from functools import wraps, partial
from torchvision.transforms import functional as F
from torch.profiler import profile, record_function, ProfilerActivity

from torchmetrics.functional.multimodal import clip_score
from torchmetrics.image.fid import FrechetInceptionDistance

from modules import memmon
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DiTPipeline, DPMSolverMultistepScheduler


def main(model_pipe=None):
    # 模型设置与加载

    device = torch.device("cuda")
    mem_mon = memmon.MemUsageMonitor('Text2Img-Thread', device) # 开启MemUsageMonitor，记录的是当前进程
    mem_mon.monitor()

    t = time.perf_counter()

    if model_pipe is None:
        model_pipe = StableDiffusionPipeline.from_single_file('../stable-diffusion-webui/models/Stable-diffusion/v2-1_768-ema-pruned.safetensors', torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        model_pipe.to("cuda")
    
    print(f"Load Model Time:{load_model_time(t)}")

    # 推理基本参数设置

    prompt = "An astronaut riding a green horse"
    height = 1024
    width = 1024
    steps = 20

    # 单次推理简要数据统计（第一个阶段）
    inference_single_time(model_pipe, mem_mon, prompt, height, width, steps)

    # 详细性能记录（第二个阶段）
    performance_record(model_pipe, prompt, height, width, steps)

    # 生图质量评估（第三个阶段）
    evaluation(model_pipe, height, width, steps)


def load_model_time(t):
    # 模型加载过程
    elapsed = time.perf_counter() - t
    elapsed_m = int(elapsed // 60)
    elapsed_s = elapsed % 60
    elapsed_text = f"{elapsed_s:.1f} sec."
    if elapsed_m > 0:
        elapsed_text = f"{elapsed_m} min. " + elapsed_text
    return elapsed_text

def calculate_clip_score(clip_score_fn, images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

def download(url, local_filepath):
    r = requests.get(url)
    with open(local_filepath, "wb") as f:
        f.write(r.content)
    return local_filepath

def preprocess_image(image):
        image = torch.tensor(image).unsqueeze(0)
        image = image.permute(0, 3, 1, 2) / 255.0
        return F.center_crop(image, (256, 256))

def inference_single_time(pipe, mem_mon, prompt, height, width, steps):
    '''
    第一阶段

    用于单次推理的简单数值记录, 包括推理时间, 显存实际占用峰值, 分配显存实际占用峰值（Pytorch有相应的Reserved机制, 会占用部分分配的显存但不释放）
    '''

    t = time.perf_counter()
    image = pipe(prompt=prompt, height=height, width=width, num_inference_steps=steps).images[0]

    # 模型单次推理时间
    print(f"Inference Time:{load_model_time(t)}")

    # 简要GPU监控记录
    mem_stats = {k: -(v//-(1024*1024)) for k, v in mem_mon.stop().items()}
    active_peak = mem_stats['active_peak']
    active = mem_stats['active']
    reserved_peak = mem_stats['reserved_peak']
    reserved = mem_stats['reserved']

    toltip_a = "Active: peak amount of video memory used during generation (excluding cached data)"
    toltip_r = "Reserved: total amount of video memory allocated by the Torch library "

    print(toltip_a)
    print(f"Inference Active peak: {active_peak:.2f} MiB")
    print(toltip_r)
    print(f"Inference Reserved peak: {reserved_peak:.2f} MiB")

    # 保存生成的图片到本地
    image.save("./image_output/output_image.jpg") # 这里的保存路径请自行设置


def performance_record(pipe, prompt, height, width, steps):
    '''
    第二阶段

    对推理过程性能开销进行全覆盖的统计, 基于Pytorch.Profiler实现, 非常全面, 支持TensorBoard查看

    这里默认已经执行完第一阶段, 所以将对应的预热阶段忽略了
    如果并没有进行模型预热, 请不要在schedule中设置对应的warmup（这是支持多轮训练中用的）
    请直接补充对应的循环对模型进行预热
    '''

    log_dir = "./log/StableDiffusionV1-5_Copy" #这里请修改为你自己的存放目录，用于存储生成的json文件


    '''
    # Warm-up
    for _ in range(3):
        start = time.time()
        images = pipe(prompt=prompt, height=height, width=width, num_inference_steps=20).images
        torch.cuda.synchronize()
        end = time.time()
        print('Time:{}ms'.format((end-start)*1000))
    '''

    with profile(
       activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        use_cuda=True,
        profile_memory=True,
        with_stack=True,
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir) 
    ) as prof:
        with record_function("model_inference"):
            images = pipe(prompt=prompt, height=height, width=width, num_inference_steps=20).images

def evaluation(pipe, height, width, steps):
    '''
    第三阶段

    对生图的效果进行量化评估, 借助于CLIP Score和FID这两个分数, 可以作为参考

    这里面Prompts, 以及计算FID对比使用的Real Images都支持自定义
    '''

    #首先计算CLIP Score（这里如果要对比两个模型的CLIP Score请把Seeds和Generate固定）

    prompts = [
    "a photo of an astronaut riding a horse on mars",
    "A high tech solarpunk utopia in the Amazon rainforest",
    "A pikachu fine dining with a view to the Eiffel Tower",
    "A mecha robot in a favela in expressionist style",
    "an insect robot preparing a delicious meal",
    "A small cabin on top of a snowy mountain in the style of Disney, artstation",
    ]

    images = pipe(prompt=prompts, num_images_per_prompt=1, height=height, width=width, num_inference_steps=steps, output_type="np").images

    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
    sd_clip_score = calculate_clip_score(clip_score_fn, images, prompts)

    print(f"CLIP score: {sd_clip_score}")

    # 计算FID

    # 这里必须使用Class-conditional image generation model, 目前还不支持任意自定义的模型（待解决）

   '''
    local_filepath='./sample-imagenet-images.zip'

    with ZipFile(local_filepath, "r") as zipper:
        zipper.extractall(".")

    dataset_path = "sample-imagenet-images"
    image_paths = sorted([os.path.join(dataset_path, x) for x in os.listdir(dataset_path)])
    real_images = [np.array(Image.open(path).convert("RGB")) for path in image_paths]

    real_images = torch.cat([preprocess_image(image) for image in real_images])
    print(real_images.shape)

    words = [ #这个给定的测试集对应的prompts
    "cassette player",
    "chainsaw",
    "chainsaw",
    "church",
    "gas pump",
    "gas pump",
    "gas pump",
    "parachute",
    "parachute",
    "tench",
    ]

    output  = pipe(prompt=words, num_images_per_prompt=1, height=height, width=width, num_inference_steps=20).images

    fake_images = output
    fake_images = torch.tensor(fake_images)
    fake_images = fake_images.permute(0, 3, 1, 2)

    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

    print(f"FID: {float(fid.compute())}")
    '''   



if __name__ == "__main__":
    main()