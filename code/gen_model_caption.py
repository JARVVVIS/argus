import os
import json
import argparse

from configs import ROOT_DIR, MODEL_MODULES, get_model_module


def wrapper(args, model_module, model_dict):
    model_name = (
        args.model_name.split("/")[-1].replace(".", "_").replace("-", "_").lower()
    )
    args.clip_path = f"{ROOT_DIR}/assets/clips/" + args.clip_name + ".mp4"
    assert os.path.isfile(args.clip_path), f"Video not found at {args.clip_path}"

    if args.model_temp is None:
        save_dir = (
            f"{ROOT_DIR}/assets/captions_{args.num_frames}/"
            + args.clip_path.split("/")[-1].split(".")[0]
            + "/"
        )
    else:
        save_dir = (
            f"{ROOT_DIR}/assets/captions_{args.num_frames}_Temp{args.model_temp}/"
            + args.clip_path.split("/")[-1].split(".")[0]
            + "/"
        )

    os.makedirs(save_dir, exist_ok=True)
    model_name = (
        model_name.replace("/", "_")
        .replace(".", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )
    save_path = save_dir + f"{model_name}.json"

    if os.path.isfile(save_path) and not args.overwrite:
        print(f"Caption already exists at {save_path}")
        return

    ## Get video specific prompt
    prompt = "Describe the video in great detail."

    ## Get Model Response
    output = model_module.generate_caption(args, model_dict, prompt)

    print(f"Caption: {output}")

    if args.verbose:
        print("-" * 50)
        print(f"Model: {model_name}")
        print(f"Caption: {output}")
        print("-" * 50)

    json_dict = {"prompt": prompt, "caption": output}
    with open(save_path, "w") as f:
        json.dump(json_dict, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--clip_name",
        type=str,
        default=None,
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--model_temp", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    models_to_run = [args.model_name] if args.model_name else MODEL_MODULES.keys()

    clips_to_run = (
        [args.clip_name] if args.clip_name else os.listdir(f"{ROOT_DIR}/assets/clips/")
    )

    for model_name in models_to_run:
        args.model_name = model_name

        try:
            model_module = get_model_module(model_name)
        except Exception as e:
            print(f"Error in loading model {model_name} => {e}")
            continue

        model_dict = model_module.load_model(args)
        for clip_name in clips_to_run:
            args.clip_name = clip_name.split(".")[0]
            try:
                wrapper(args, model_module, model_dict)
            except Exception as e:
                print(
                    f"Error in generating caption for {model_name} and {args.clip_name} => {e}"
                )
                continue


if __name__ == "__main__":
    main()
