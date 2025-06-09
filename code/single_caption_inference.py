## standalone file to calculate the metrics given a single (human caption, model caption) pair

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
from nli_eval import get_base_prompt, generate_and_parse_response
from compute_metrics import TargetNode, DPHallucinationMeasure
from configs import ROOT_DIR, get_judge_module, JUDGE_MODULES, MODEL_MODULES


def get_prompt(args, human_caption, model_caption):

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
    prompt,
    human_caption,
    model_caption,
    judge_setup,
    judge_module,
):

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

    return result_file


def get_nli_judgement(args, judge_setup, judge_module, human_caption, model_caption):

    ## Hallucination Analysis
    args.analysis_mode = "hallucination"
    args.reference_model = "human"
    ## get prompt
    hall_prompt, human_caption, model_caption = get_prompt(
        args, human_caption, model_caption
    )
    hall_result_file = generate_and_parse_response(
        args,
        hall_prompt,
        human_caption,
        model_caption,
        judge_setup,
        judge_module,
    )

    ## Omission Analysis
    args.analysis_mode = "omission"
    args.target_model = "human"
    ## get prompt
    omm_prompt, human_caption, model_caption = get_prompt(
        args, human_caption, model_caption
    )

    omm_result_file = generate_and_parse_response(
        args,
        omm_prompt,
        human_caption,
        model_caption,
        judge_setup,
        judge_module,
    )

    return hall_result_file, omm_result_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip_name",
        type=str,
        default="mailman",
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--judge_model", type=str, default="gpt-4o", choices=JUDGE_MODULES.keys()
    )
    parser.add_argument("--judge_temp", type=float, default=0.0)
    parser.add_argument(
        "--model_caption",
        type=str,
        default=None,
        help="can be the path or the caption directly",
        required=True,
    )
    parser.add_argument("--order_penalty_fac", type=float, default=0.1)

    args = parser.parse_args()

    ## judge stuff
    judge_module = get_judge_module(args.judge_model)
    judge_setup = judge_module.judge_setup(args)

    ## load the DF
    argus_df = load_dataset("RuchitRawal/argus", split="train").to_pandas()
    human_caption = argus_df[(argus_df["clip_name"] == args.clip_name)][
        "human_caption"
    ].values[0]

    if os.path.isfile(args.model_caption):
        ## assuming it's a .txt file
        with open(args.model_caption, "r") as f:
            model_caption = f.read()
    else:
        ## assuming it's a caption
        model_caption = args.model_caption.strip()

    clip_name = args.clip_name.split(".")[0]
    try:
        hall_result_file, omm_result_file = get_nli_judgement(
            args, judge_setup, judge_module, human_caption, model_caption
        )
    except Exception as e:
        print(f"\tError in processing {clip_name} ==> {e}")
        return

    ## compute metrics -- hallucination
    hall_dp_measure = DPHallucinationMeasure(
        hall_result_file,
        order_penalty_fac=args.order_penalty_fac,
        mode="hallucination",
        verbose=False,
    )
    hall_curr_result = hall_dp_measure.compute_optimal_matching()
    print(f'\tHallucination-Cost: {hall_curr_result["normalized_cost"]*100:.2f}%')

    omm_dp_measure = DPHallucinationMeasure(
        omm_result_file,
        order_penalty_fac=args.order_penalty_fac,
        mode="omissions",
        verbose=False,
    )
    omm_curr_result = omm_dp_measure.compute_optimal_matching()
    print(f'\Omission-Cost: {omm_curr_result["normalized_cost"]*100:.2f}%')


if __name__ == "__main__":
    main()
