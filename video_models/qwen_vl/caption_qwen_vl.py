import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
import argparse
import cv2


def load_model(args):

    if args.model_name == "qwen2_5_vl_7b_instruct":
        pretrained = "Qwen/Qwen2.5-VL-7B-Instruct"
    elif args.model_name == "qwen2_5_vl_7b_instruct_awq":
        pretrained = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
    elif args.model_name == "qwen2_5_vl_3b_instruct":
        pretrained = "Qwen/Qwen2.5-VL-3B-Instruct"
    elif args.model_name == "qwen2_5_vl_3b_instruct_awq":
        pretrained = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
    else:
        raise NotImplementedError(f"Model {args.model_name} not implemented")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        pretrained,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(pretrained)

    return {"model": model, "processor": processor}


def get_desired_fps_given_video_path_and_num_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps * num_frames / num_frames_in_video


def generate_caption(args, model_dict, prompt="Describe the video in great detail"):

    model, processor = model_dict["model"], model_dict["processor"]

    desired_fps = get_desired_fps_given_video_path_and_num_frames(
        args.clip_path, args.num_frames
    )

    print(f"Using FPS: {desired_fps}")

    # Messages containing a video and a text query
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": args.clip_path,
                    "max_pixels": 360 * 420,
                    "fps": desired_fps,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(
        **inputs, max_new_tokens=1000, temperature=args.model_temp
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    ## user prompt decode
    user_ids = [
        out_ids[: len(in_ids)]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    user_text = processor.batch_decode(
        user_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print("User prompt:", user_text[0])

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen2_5_vl_7b_instruct",
    )
    parser.add_argument(
        "--clip_name",
        type=str,
        default="trimmed_chameleon.mp4",
    )
    parser.add_argument("--num_frames", type=int, default=16)
    args = parser.parse_args()

    args.clip_path = args.clip_name

    model_dict = load_model(args)
    caption = generate_caption(args, model_dict)
    print(caption)


if __name__ == "__main__":
    main()
