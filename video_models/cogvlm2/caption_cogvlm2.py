import io
import numpy as np
import torch
from decord import cpu, VideoReader, bridge
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse

import os
import tqdm
import subprocess
from datasets import load_dataset


def load_video(video_path, strategy="chat"):
    bridge.set_bridge("torch")
    with open(video_path, "rb") as f:
        mp4_stream = f.read()
    num_frames = 24

    if mp4_stream is not None:
        decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0), num_threads=1)
    else:
        decord_vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    frame_id_list = None
    total_frames = len(decord_vr)
    if strategy == "base":
        clip_end_sec = 60
        clip_start_sec = 0
        start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
        end_frame = (
            min(total_frames, int(clip_end_sec * decord_vr.get_avg_fps()))
            if clip_end_sec is not None
            else total_frames
        )
        frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    elif strategy == "chat":
        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
        timestamps = [i[0] for i in timestamps]
        max_second = round(max(timestamps)) + 1
        frame_id_list = []
        for second in range(max_second):
            closest_num = min(timestamps, key=lambda x: abs(x - second))
            index = timestamps.index(closest_num)
            frame_id_list.append(index)
            if len(frame_id_list) >= num_frames:
                break
    video_data = decord_vr.get_batch(frame_id_list)
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data


def load_model(args):

    if args.model_name == "cogvlm2":
        pretrained = "THUDM/cogvlm2-video-llama3-chat"
    else:
        raise NotImplementedError(f"Model {args.model_name} not implemented")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_type = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained,
        trust_remote_code=True,
        # padding_side="left"
    )

    if args.quant == 4:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            torch_dtype=torch_type,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_type,
            ),
            low_cpu_mem_usage=True,
            cache_dir=f"./cache/",
        ).eval()
    elif args.quant == 8:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            torch_dtype=torch_type,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_compute_dtype=torch_type,
            ),
            low_cpu_mem_usage=True,
            cache_dir=f"./cache/",
        ).eval()
    else:
        model = (
            AutoModelForCausalLM.from_pretrained(
                pretrained,
                torch_dtype=torch_type,
                trust_remote_code=True,
                cache_dir=f"./cache/",
            )
            .eval()
            .to(device)
        )

    strategy = "base" if "cogvlm2-video-llama3-base" in pretrained else "chat"

    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "torch_type": torch_type,
        "strategy": strategy,
    }


def generate_caption(args, model_dict, prompt="Describe the video in great detail"):
    model, tokenizer, device, torch_type, strategy = (
        model_dict["model"],
        model_dict["tokenizer"],
        model_dict["device"],
        model_dict["torch_type"],
        model_dict["strategy"],
    )

    video = load_video(args.clip_path, strategy=strategy)
    inp_prompt = f"Human: {prompt}\n"

    history = []
    inputs = model.build_conversation_input_ids(
        tokenizer=tokenizer,
        query=inp_prompt,
        images=[video],
        history=history,
        template_version=strategy,
    )

    inputs = {
        "input_ids": inputs["input_ids"].unsqueeze(0).to(device),
        "token_type_ids": inputs["token_type_ids"].unsqueeze(0).to(device),
        "attention_mask": inputs["attention_mask"].unsqueeze(0).to(device),
        "images": [[inputs["images"][0].to("cuda").to(torch_type)]],
    }
    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
        "top_k": 1,
        "do_sample": True,
        "top_p": 0.1,
        "temperature": 0.1,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


def main():
    parser = argparse.ArgumentParser(description="CogVLM2-Video CLI Demo")
    parser.add_argument(
        "--model_name",
        type=str,
        default="cogvlm2",
    )
    parser.add_argument(
        "--clip_name",
        type=str,
        default="trimmed_chameleon.mp4",
    )
    parser.add_argument("--num_frames", type=int, default=16)
    args = parser.parse_args()

    args.clip_path = args.clip_name
    args.quant = 4

    model_dict = load_model(args)
    caption = generate_caption(args, model_dict)

    print(f"Response: {caption}")


if __name__ == "__main__":
    main()
