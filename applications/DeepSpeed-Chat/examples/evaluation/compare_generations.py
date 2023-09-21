# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import deepspeed
import logging
import torch
import sys
import os
import json

from transformers import (
    AutoTokenizer
)

from deepspeed_chat.utils.model.model_utils import create_critic_model
from deepspeed_chat.utils.ds_utils import get_eval_ds_config
from rouge_score import rouge_scorer


logger = logging.getLogger(__name__)


def load_jsonl(file):
    results = []
    num_results = 0
    with open(file, "r") as fp:
        for line in fp:
            if any(not x.isspace() for x in line):
                results.append(json.loads(line))
                num_results += 1

    print("num jsonl lines", num_results)
    return results


def get_reward_score(tokenizer, prompt_token_len, generate_ids, reward_model):
    pad_token_id = tokenizer.pad_token_id
    prompt_length = prompt_token_len
    seq = generate_ids

    attention_mask = seq.not_equal(pad_token_id).long()

    r_seq = seq.to(reward_model.v_head.weight.device)
    r_attention_mask = attention_mask.to(reward_model.v_head.weight.device)
    reward_score = reward_model.forward_value(r_seq, r_attention_mask, prompt_length=prompt_length)['chosen_end_scores'].detach().flatten().item()

    return reward_score


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")

    parser.add_argument(
        "--baseline_model_outputs",
        type=str,
        help="Path to baseline model outputs",
        required=False,
    )
    parser.add_argument(
        "--finetune_model_outputs",
        type=str,
        help="Path to baseline model outputs",
        required=False,
    )
    parser.add_argument(
        "--reward_model_path",
        type=str,
        help="Path to reward model",
        required=False,
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help="OPT model has a fixed number (1) of padding tokens at the beginning of the input. We did not see this in other models but keep it as an option for now."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the outputs."
    )

    args = parser.parse_args()

    return args

def print_rouge_stats(rouge_scores):
    for key in ["precision", "recall", "fmeasure"]:
        rouge1_scores = [getattr(x["rouge1"], key) for x in rouge_scores]
        rougeL_scores = [getattr(x["rougeL"], key) for x in rouge_scores]

        print("ROUGE 1", key, mean(rouge1_scores), "std", std(rouge1_scores))

    print()

    for key in ["precision", "recall", "fmeasure"]:
        rouge1_scores = [getattr(x["rouge1"], key) for x in rouge_scores]
        rougeL_scores = [getattr(x["rougeL"], key) for x in rouge_scores]
        print("ROUGE L", key, mean(rougeL_scores), "std", std(rougeL_scores))

def mean(items):
    if len(items) > 0:
        return sum(items) / len(items)

    print("0-length list given to mean, returning -1...")
    return -1

def std(items, this_mean=None):
    if this_mean is None:
        this_mean = sum(items) / len(items)
        variance = sum([((x - this_mean) ** 2) for x in items]) / len(items)
        res = variance ** 0.5
        return res
    else:
        variance = sum([((x - this_mean) ** 2) for x in items]) / len(items)
        res = variance ** 0.5
        return res

def check_dir_get_file(path):
    if os.path.isdir(path):
        outputs = os.listdir(path)
        assert len(outputs) == 1
        return os.path.join(path, outputs[0])
    else:
        return path


def main():
    args = parse_args()

    if "LOCAL_RANK" in os.environ:
        local_rank = os.environ.get('LOCAL_RANK')
        args.local_rank = int(local_rank)
        print("My local rank is", args.local_rank)

        torch.cuda.set_device(args.local_rank)

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    device = torch.device("cuda:0")

    args.zero_stage = 3

    print("Using zero stage", args.zero_stage)

    ds_config = get_eval_ds_config(offload=False, stage=args.zero_stage)

    if args.zero_stage == 3:
        loading_ds_config = get_eval_ds_config(offload=False, stage=0)   # necessary JUST to load in model without shard issues
    else:
        loading_ds_config = ds_config

    baseline_model_outputs_list = load_jsonl(check_dir_get_file(args.baseline_model_outputs))
    finetune_model_outputs_list = load_jsonl(check_dir_get_file(args.finetune_model_outputs))

    baseline_model_outputs = {
        item["prompt"]: item for item in baseline_model_outputs_list
    }

    finetune_model_outputs = {
        item["prompt"]: item for item in finetune_model_outputs_list
    }

    prompts = set(baseline_model_outputs.keys()).intersection(set(finetune_model_outputs.keys()))

    if len(prompts) != len(baseline_model_outputs.keys()) or len(prompts) != len(finetune_model_outputs.keys()):
        print("There are not matching number of evaluation prompts")

        print("baseline prompts", len(baseline_model_outputs.keys()))
        print("finetune prompts", len(finetune_model_outputs.keys()))
        print("intersect", len(prompts))

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    baseline_rouge_scores = []
    finetune_rouge_scores = []

    tokenizer = AutoTokenizer.from_pretrained(
        args.reward_model_path,
        fast_tokenizer=True,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        truncation_side="left",
        padding_side="right"  # necessary for llama
    )
    tokenizer.pad_token = tokenizer.eos_token

    reward_model=None
    if args.reward_model_path is not None and os.path.isdir(args.reward_model_path):
        print("loading reward model")
        reward_model = create_critic_model(
            model_name_or_path=args.reward_model_path,
            tokenizer=tokenizer,
            ds_config=None,
            # ds_config=loading_ds_config,
            num_padding_at_beginning=args.num_padding_at_beginning,
            disable_dropout=True,
            rlhf_training=True,
        )
        reward_model.to("cuda:1")
        reward_model.eval()

    total_baseline_reward = 0
    total_finetuned_reward = 0

    base_reward_wins = 0
    finetune_reward_wins = 0
    reward_ties = 0

    base_rouge_wins = 0
    finetune_rouge_wins = 0
    rouge_ties = 0

    baseline_lens = []
    finetune_lens = []

    results = []
    for prompt in prompts:
        baseline = baseline_model_outputs[prompt]
        finetune = finetune_model_outputs[prompt]

        baseline_cleaned = baseline["response"]
        finetune_cleaned = finetune["response"]
        baseline_generate_ids = torch.tensor(baseline["generate_ids"])
        finetune_generate_ids = torch.tensor(finetune["generate_ids"])

        prompt_token_len = baseline["prompt_token_len"]

        baseline_rouge_score = scorer.score(prompt, baseline_cleaned)
        baseline_rouge_scores.append(baseline_rouge_score)
        
        finetune_rouge_score = scorer.score(prompt, finetune_cleaned)
        finetune_rouge_scores.append(finetune_rouge_score)

        baseline_lens.append(len(baseline_cleaned))
        finetune_lens.append(len(finetune_cleaned))

        with torch.no_grad():
            if reward_model is not None:
                baseline_reward_score = get_reward_score(tokenizer, prompt_token_len, baseline_generate_ids, reward_model)
            else:
                baseline_reward_score = 0

            if reward_model is not None:
                finetuned_reward_score = get_reward_score(tokenizer, prompt_token_len, finetune_generate_ids, reward_model)
            else:
                finetuned_reward_score = 0

        total_baseline_reward += baseline_reward_score
        total_finetuned_reward += finetuned_reward_score

        if finetuned_reward_score == baseline_reward_score:
            reward_ties += 1
        elif finetuned_reward_score > baseline_reward_score:
            finetune_reward_wins += 1
        else:
            base_reward_wins += 1


        if finetune_rouge_score["rougeL"].fmeasure == baseline_rouge_score["rougeL"].fmeasure:
            rouge_ties += 1
        elif finetune_rouge_score["rougeL"].fmeasure > baseline_rouge_score["rougeL"].fmeasure:
            finetune_rouge_wins += 1
        else:
            base_rouge_wins += 1

            
        print("REWARD SCORES")
        print("baseline_reward_score", baseline_reward_score)
        print("finetuned_reward_score", finetuned_reward_score)
        print("base_reward_wins", base_reward_wins)
        print("finetune_reward_wins", finetune_reward_wins)
        print("baseline_rouge_score", baseline_rouge_score)
        print("finetune_rouge_score", finetune_rouge_score)
        print("base_rouge_wins", base_rouge_wins)
        print("finetune_rouge_wins", finetune_rouge_wins)
        print()
        print()

        results.append({
            "prompt": prompt,
            "baseline": baseline_cleaned,
            "finetuned": finetune_cleaned,
        })

    print()
    print()
    print()
    print()
    print()
    print("REWARD MODEL WINS")

    print("base_reward_wins", base_reward_wins, base_reward_wins/len(prompts))
    print("finetune_reward_wins", finetune_reward_wins, finetune_reward_wins/len(prompts))
    print("reward_ties", reward_ties, reward_ties/len(prompts))
    print()
    print("baseline lens", mean(baseline_lens), "std", std(baseline_lens))
    print("finetune lens", mean(finetune_lens), "std", std(finetune_lens))
    print()
    print("base_rouge_wins", base_rouge_wins, base_rouge_wins/len(prompts))
    print("finetune_rouge_wins", finetune_rouge_wins, finetune_rouge_wins/len(prompts))

    print()
    print("ROUGE SCORES")
    print()
    print("BASELINE")
    print_rouge_stats(baseline_rouge_scores)
    print()
    print("FINETUNE")
    print_rouge_stats(finetune_rouge_scores)
    print()

    with open(os.path.join(args.output_dir, "outputs.jsonl"), 'w') as fp:
        for line in results:
            fp.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
