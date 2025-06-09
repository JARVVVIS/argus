import json
import os
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset

import numpy as np

from configs import ROOT_DIR
from utils.line_by_line_utils import (
    parse_structured_text,
)

from configs import ROOT_DIR, get_judge_module, JUDGE_MODULES, MODEL_MODULES


def get_base_prompt(args):

    ## Load the base prompt
    base_prompt_path = f"{ROOT_DIR}/assets/prompts/base_prompt.txt"
    with open(base_prompt_path, "r") as f:
        base_prompt = f.read()

    assert base_prompt is not None
    assert "{IN_CONTEXT_EXAMPLES}" in base_prompt
    assert "{source_caption}" in base_prompt
    assert "{target_caption}" in base_prompt

    return base_prompt


def get_prompt(args, argus_df):
    row = argus_df[(argus_df["clip_name"] == args.clip_name)]
    assert len(row) == 1
    human_caption = row.iloc[0]["human_caption"]

    model_caption_path = f"{ROOT_DIR}/assets/captions_{args.num_frames}/{args.clip_name}/{args.model_name}.json"  ## load "caption" feild

    with open(model_caption_path, "r") as f:
        model_caption = json.load(f)["caption"]

    base_prompt = get_base_prompt(args)

    ## setup in-context examples
    in_context_dir = f"{ROOT_DIR}/assets/prompts/in_context_examples"
    in_context_files = os.listdir(in_context_dir)

    ## if args.clip_name.txt is present, use that as the in-context example
    if args.clip_name + ".txt" in in_context_files:
        in_context_files = [f"{args.clip_name}.txt"]
    ## if not, randomly select 3 examples
    else:
        in_context_files = np.random.choice(in_context_files, 3, replace=False)

    ## build the string
    in_context_str = ""

    for in_context_file in in_context_files:
        with open(os.path.join(in_context_dir, in_context_file), "r") as f:
            in_context_str += f.read() + "\n\n"

    base_prompt = base_prompt.replace("{IN_CONTEXT_EXAMPLES}", in_context_str)

    if args.analysis_mode == "hallucination":
        prompt = base_prompt.replace("{source_caption}", human_caption).replace(
            "{target_caption}", model_caption
        )
    elif args.analysis_mode == "omission":
        prompt = base_prompt.replace("{source_caption}", model_caption).replace(
            "{target_caption}", human_caption
        )  ## source is model; and target human

    assert "{IN_CONTEXT_EXAMPLES}" not in prompt
    assert "{source_caption}" not in prompt
    assert "{target_caption}" not in prompt

    return prompt, human_caption, model_caption


def generate_and_parse_response(
    args,
    save_evals_path,
    prompt,
    human_caption,
    model_caption,
    judge_setup,
    judge_module,
):
    if not os.path.isfile(save_evals_path) or args.overwrite:
        print(f"\tProcessing: {save_evals_path}")

        eval_output = judge_module.judge_response(prompt, judge_setup)

        results = parse_structured_text(eval_output)

        assert results is not None
        assert len(results) > 0

        ## unit-tests for results

        # 6 elements in each tuple
        assert all(len(t) == 6 for t in results)
        # 0th-element int, rest all strings
        assert all(isinstance(t[0], int) for t in results)
        assert all(isinstance(t[1], str) for t in results)

        # last element is one of: "entailment", "plausible", "implausible"
        assert all(
            t[5] in ["entailment", "contradiction", "underdetermined"] for t in results
        )

        # 3rd index is one of: "visual-description", "summary", "dynamic-action"
        assert all(
            t[3] in ["visual-description", "summary", "dynamic-action"] for t in results
        )

        # ## save results
        result_file = {
            "analysis_mode": args.analysis_mode,
            "human_caption": human_caption,
            "model_caption": model_caption,
            "result_raw": eval_output,
            "result_tuple": results,
        }

        with open(save_evals_path, "w") as f:
            json.dump(result_file, f, indent=4)
    else:
        print(f"\tLoading from: {save_evals_path}")
        ## Load the saved results
        with open(save_evals_path, "r") as f:
            result_file = json.load(f)

    return result_file


def wrapper(args, judge_setup, judge_module, argus_df):

    ## Hallucination Analysis
    args.analysis_mode = "hallucination"
    args.reference_model = "human"
    args.target_model = args.model_name.replace("-", "_").replace(".", "_").lower()
    save_evals_dir = f"{ROOT_DIR}/assets/evals/frames{args.num_frames}/judge_model_{args.judge_model}/{args.clip_name}/REF_{args.reference_model}"
    os.makedirs(save_evals_dir, exist_ok=True)

    ## get prompt
    hall_prompt, human_caption, model_caption = get_prompt(args, argus_df)

    save_evals_path = f"{save_evals_dir}/TAR_{args.target_model}.json"
    hall_result_file = generate_and_parse_response(
        args,
        save_evals_path,
        hall_prompt,
        human_caption,
        model_caption,
        judge_setup,
        judge_module,
    )

    ## Omission Analysis
    args.analysis_mode = "omission"
    args.reference_model = args.model_name.replace("-", "_").replace(".", "_").lower()
    args.target_model = "human"
    save_evals_dir = f"{ROOT_DIR}/assets/evals/frames{args.num_frames}/judge_model_{args.judge_model}/{args.clip_name}/REF_{args.reference_model}"
    os.makedirs(save_evals_dir, exist_ok=True)
    save_evals_path = f"{save_evals_dir}/TAR_{args.target_model}.json"

    ## get prompt
    omm_prompt, human_caption, model_caption = get_prompt(args, argus_df)

    omm_result_file = generate_and_parse_response(
        args,
        save_evals_path,
        omm_prompt,
        human_caption,
        model_caption,
        judge_setup,
        judge_module,
    )


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
    parser.add_argument("--overwrite", action="store_true", default=False)

    parser.add_argument(
        "--judge_model", type=str, default="gpt-4o", choices=JUDGE_MODULES.keys()
    )
    parser.add_argument("--judge_temp", type=float, default=0.0)
    args = parser.parse_args()

    ## judge stuff
    judge_module = get_judge_module(args.judge_model)
    judge_setup = judge_module.judge_setup(args)

    ## load the DF
    argus_df = load_dataset("tomg-group-umd/argus", split="train").to_pandas()

    models_to_run = [args.model_name] if args.model_name else MODEL_MODULES.keys()
    clips_to_run = (
        [args.clip_name]
        if args.clip_name
        else os.listdir(f"{ROOT_DIR}/assets/clip-bank/")
    )
    for model_name in models_to_run:
        for clip_idx, clip_name in enumerate(clips_to_run):
            print(
                f"Processing: {model_name} for {clip_name} ({clip_idx+1}/{len(clips_to_run)})"
            )

            args.model_name = (
                model_name.replace("_", "-")
                .replace(".", "_")
                .replace("-", "_")
                .replace(" ", "_")
            )
            args.clip_name = clip_name.split(".")[0]
            try:
                wrapper(args, judge_setup, judge_module, argus_df)
            except Exception as e:
                print(
                    f"\tError in processing {args.model_name} for {clip_name} ==> {e}"
                )
                continue


if __name__ == "__main__":
    main()
