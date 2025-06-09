import os
import json
import argparse
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


from configs import ROOT_DIR, JUDGE_MODULES, MODEL_MODULES
from utils import (
    split_into_sentences,
    find_evidence_line_number,
    find_evidence_line_number,
    filter_content_line,
    compute_words_in_text,
)


## Class to store the target node information
@dataclass
class TargetNode:
    line_num: int
    line_txt: str
    evidence: str
    type_: str
    reasoning: str
    verdict: str
    evidence_line_num: int = None

    def set_evidence_line_num(self, source_caption):
        self.evidence_line_num = find_evidence_line_number(
            self.evidence, source_caption
        )


## DP-Framework
class DPHallucinationMeasure:
    def __init__(
        self,
        result_file: Dict,
        order_penalty_fac: float = 1.0,
        mode="hallucination",
        verbose=False,
    ):
        self.result_file = result_file
        self.source_caption = (
            result_file["human_caption"]
            if mode == "hallucination"
            else result_file["model_caption"]
        )
        self.source_lines = split_into_sentences(self.source_caption)
        self.order_penalty_fac = order_penalty_fac
        self.verbose = verbose

        # Initialize target nodes
        self.target_nodes = self._create_target_nodes()

        self.num_targets = len(self.target_nodes)
        self.num_sources = len(self.source_lines)
        if self.verbose:
            print(f"{self.num_targets} Target Lines & {self.num_sources} Source Lines!")

        # Precompute base cost matrix
        self.base_cost_matrix = self._compute_base_cost_matrix()

        self.dp_table = np.full((self.num_targets + 1, self.num_sources), np.inf)
        # Initialize with empty lists properly
        self.optimal_paths = np.empty(
            (self.num_targets + 1, self.num_sources), dtype=object
        )
        for i in range(self.num_targets + 1):
            for j in range(self.num_sources):
                self.optimal_paths[i, j] = []

        self.order_penalty_fac = order_penalty_fac

    def _create_target_nodes(self) -> List[TargetNode]:
        target_nodes = []

        did_filter = 0

        for res in self.result_file["result_tuple"]:
            line_num, line_txt, evidence, type_, reasoning, verdict = res

            # pre-filter the line if needed
            if not filter_content_line(line_txt):
                did_filter += 1
                target_node = TargetNode(
                    line_num,
                    "header text",
                    evidence,
                    "summary",
                    "header text",
                    "entailment",
                )
                target_node.set_evidence_line_num(self.source_caption)
                target_nodes.append(target_node)
                continue

            target_node = TargetNode(
                line_num, line_txt, evidence, type_, reasoning, verdict
            )
            target_node.set_evidence_line_num(self.source_caption)
            target_nodes.append(target_node)

        assert (
            did_filter / len(self.result_file["result_tuple"]) < 0.4
        ), f'Filtered too many lines: {did_filter} / {len(self.result_file["result_tuple"])}'

        return target_nodes

    def _compute_base_cost_matrix(self) -> np.ndarray:
        """Compute the base cost matrix without ordering constraints."""
        cost_matrix = np.zeros((self.num_targets, self.num_sources))
        for i, node in enumerate(self.target_nodes):
            for j in range(self.num_sources):
                if node.verdict != "entailment":
                    cost_matrix[i, j] = 1
                else:
                    if node.type_ in {"summary", "visual-description"}:
                        cost_matrix[i, j] = 0
                    else:
                        cost_matrix[i, j] = 0 if j == node.evidence_line_num else 1
        return cost_matrix

    def _compute_ordering_penalty(self, i, j, optimal_tuples):
        # Return how many values in prev_path > j
        penalty = 0
        if (
            self.target_nodes[i].type_ != "dynamic-action"
            and self.target_nodes[i].verdict != "entailment"
        ):
            ## if the current node to which we have to add cost is not an action and entailed we dont need to bother about the penalty.
            return penalty

        for prev_path_j, prev_node in optimal_tuples:
            ## we need to check for all previous nodes that were dynamic actions and entailed; see if our current node is being added before them which would be a penalty voilation.
            if (
                prev_node.type_ == "dynamic-action"
                and prev_node.verdict == "entailment"
                and prev_path_j > j
            ):
                penalty += self.order_penalty_fac

        return penalty

    def compute_optimal_matching(self):
        self.dp_table[0, :] = 0

        for i in range(self.num_targets):
            for j in range(self.num_sources):
                base_cost = self.base_cost_matrix[i, j]

                for prev_j in range(self.num_sources):

                    prev_cost = self.dp_table[
                        i, prev_j
                    ]  ## cummulative cost till previous target node
                    prev_optimal_tuples = self.optimal_paths[
                        i, prev_j
                    ]  ## optimal path till previous target node

                    if i != 0:
                        optimal_tuples = prev_optimal_tuples + [
                            (prev_j, self.target_nodes[i - 1])
                        ]
                    else:
                        optimal_tuples = prev_optimal_tuples

                    order_penalty = self._compute_ordering_penalty(i, j, optimal_tuples)

                    total_cost = prev_cost + base_cost + order_penalty

                    if total_cost < self.dp_table[i + 1, j]:
                        self.dp_table[i + 1, j] = total_cost

                        ## if we get a better cost; we should update the optimal path to reflect the node responsible + the
                        if i != 0:
                            self.optimal_paths[i + 1, j] = optimal_tuples.copy()

                        if self.verbose:
                            print(
                                f"\ttotal-cost: {total_cost:.2f} [base_cost={base_cost}, prev_best={prev_j}, prev_cost={prev_cost}, order_penalty={order_penalty}]"
                            )

        final_costs = self.dp_table[self.num_targets]
        optimal_end = np.argmin(final_costs)
        optimal_cost = final_costs[optimal_end]
        optimal_tuple = self.optimal_paths[self.num_targets, optimal_end]
        optimal_path = [path for path, _ in optimal_tuple] + [optimal_end]
        self.optimal_path = optimal_path
        

        max_dynamic_actions = sum(
            1
            for node in self.target_nodes
            if node.type_ == "dynamic-action" and node.verdict == "entailment"
        )
        max_order_penalties = (max_dynamic_actions * (max_dynamic_actions - 1)) / 2
        max_cost = (
            self.num_targets - max_dynamic_actions
        ) + self.order_penalty_fac * max_order_penalties

        if self.verbose:
            print(
                f"max made-up cost: {self.num_targets - max_dynamic_actions}, max_dynamic_actions={max_dynamic_actions}, max_order_penalties={self.order_penalty_fac * max_order_penalties}"
            )

        return {
            "optimal_cost": optimal_cost,
            "normalized_cost": (optimal_cost / max_cost),
            "optimal_path": optimal_path,
            "max_cost": max_cost,
            "max_dynamic_actions": max_dynamic_actions,
            "max_order_penalties": max_order_penalties,
            "max_made_up_cost": self.num_targets - max_dynamic_actions,
            "dp_table": self.dp_table,
            "target_nodes": self.target_nodes,
            "model_caption": self.result_file["model_caption"],
            "human_caption": self.result_file["human_caption"],
            "num_targets": self.num_targets,
            "num_sources": self.num_sources,
        }

    def print_pretty_optimal_path(self):
        for tar_idx, src_idx in enumerate(self.optimal_path):
            print(f"Target: {tar_idx} -> Source: {src_idx}")
            print(f"\tTarget-Line: {self.target_nodes[tar_idx].line_txt}")
            print(f"\tEvidence: {self.target_nodes[tar_idx].evidence}")
            print(f"\tReasoning: {self.target_nodes[tar_idx].reasoning}")
            print(f"\tType: {self.target_nodes[tar_idx].type_}")
            print(f"\tVerdict: {self.target_nodes[tar_idx].verdict}")
            print(f"\tBase-Cost: {self.base_cost_matrix[tar_idx, src_idx]}")
            print(f"\tCummulative Cost: {self.dp_table[tar_idx + 1, src_idx]}")
            print()


def get_hall_omm_df(
    model_names,
    avail_clips,
    eval_dir,
    order_penalty_fac=0.1,
):

    ## build the hallucination DF
    hallucination_list = []

    for model_name in model_names:
        model_name = (
            model_name.replace("/", "_")
            .replace("-", "_")
            .replace(".", "_")
            .replace(" ", "_")
        )

        for clip in avail_clips:
            clip_name = clip.split(".")[0]

            clip_result_path = f"{eval_dir}/{clip_name}/REF_human/TAR_{model_name}.json"

            try:
                ## load the json file
                with open(clip_result_path) as f:
                    clip_result = json.load(f)

                dp_measure = DPHallucinationMeasure(
                    clip_result,
                    order_penalty_fac=order_penalty_fac,
                    mode="hallucination",
                    verbose=False,
                )
                curr_result = dp_measure.compute_optimal_matching()

                hallucination_list.append(
                    {
                        "model_name": model_name,
                        "clip": clip,
                        **curr_result,
                    }
                )
            except Exception as e:
                print(f"Error processing {model_name} for {clip} frames")
                print(e)
                ## print the traceback
                import traceback
                traceback.print_exc()
                continue

    hallucination_df = pd.DataFrame(hallucination_list)

    ## omission df
    omission_list = []

    for model_name in model_names:
        model_name = (
            model_name.replace("/", "_")
            .replace("-", "_")
            .replace(".", "_")
            .replace(" ", "_")
        )

        for clip in avail_clips:
            clip_name = clip.split(".")[0]

            clip_result_path = f"{eval_dir}/{clip_name}/REF_{model_name}/TAR_human.json"

            try:
                ## load the json file
                with open(clip_result_path) as f:
                    clip_result = json.load(f)

                dp_measure = DPHallucinationMeasure(
                    clip_result,
                    order_penalty_fac=order_penalty_fac,
                    mode="omissions",
                    verbose=False,
                )
                curr_result = dp_measure.compute_optimal_matching()

                omission_list.append(
                    {
                        "model_name": model_name,
                        "clip": clip,
                        **curr_result,
                    }
                )

            except Exception as e:
                print(f"Error processing {model_name} for {clip}")
                print(e)
                continue

    omission_df = pd.DataFrame(omission_list)

    print(f"Finished processing")
    print(
        f"Length of hallucination frame df: {len(hallucination_df)} [Unique Models: {hallucination_df['model_name'].nunique()} | Unique Clips: {hallucination_df['clip'].nunique()}]"
    )  ## contains result for all models
    print(
        f"Length of omission frame df: {len(omission_df)} [Unique Models: {hallucination_df['model_name'].nunique()} | Unique Clips: {hallucination_df['clip'].nunique()}]"
    )  ## contains result for all models

    return hallucination_df, omission_df


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
    parser.add_argument(
        "--judge_model", type=str, default="gpt-4o", choices=JUDGE_MODULES.keys()
    )
    parser.add_argument("--order_penalty_fac", type=float, default=0.1)
    parser.add_argument("--judge_temp", type=float, default=0.0)
    args = parser.parse_args()

    ### Setup

    eval_dir = f"{ROOT_DIR}/assets/evals/frames{args.num_frames}/judge_model_{args.judge_model}"
    save_dir = f"{ROOT_DIR}/assets/computed_metrics/frames{args.num_frames}/judge_model_{args.judge_model}"
    os.makedirs(save_dir, exist_ok=True)

    if args.clip_name is None:
        avail_clips = os.listdir(eval_dir)  ## however
    else:
        avail_clips = [args.clip_name]

    model_names = [args.model_name] if args.model_name else MODEL_MODULES.keys()

    hallucination_main_df, omission_main_df = get_hall_omm_df(
        model_names,
        avail_clips,
        eval_dir,
        order_penalty_fac=0.1,
    )

    ## print final metrics
    for model_name in model_names:
        model_hall_df = hallucination_main_df[
            hallucination_main_df["model_name"] == model_name
        ]
        model_omm_df = omission_main_df[omission_main_df["model_name"] == model_name]

        print(f"Model: {model_name}")
        print(f"\tHallucination-Cost: {model_hall_df['normalized_cost'].mean()}")
        print(f"\tOmission-Cost: {model_omm_df['normalized_cost'].mean()}")

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    hallucination_main_df.to_pickle(f"{save_dir}/concatenated_hall_df_{timestamp}.pkl")
    omission_main_df.to_pickle(f"{save_dir}/concatenated_omm_df_{timestamp}.pkl")


if __name__ == "__main__":
    main()
