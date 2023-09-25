# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import logging
import torch
import sys
import os
import json
import requests

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from transformers.generation  import StoppingCriteria, StoppingCriteriaList

from deepspeed_chat.utils.model.model_utils import create_hf_model
from deepspeed_chat.utils.ds_utils import get_eval_ds_config
from datasets import load_dataset, load_from_disk
from rouge_score import rouge_scorer


logger = logging.getLogger(__name__)


# def load_jsonl(file):
#     results = []
#     num_results = 0
#     with open(file, "r") as fp:
#         for line in fp:
#             if any(not x.isspace() for x in line):
#                 results.append(json.loads(line))
#                 num_results += 1

#     print("num jsonl lines", num_results)
#     return results


def load_json(file):
    with open(file, "r") as fp:
        result = json.load(fp)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Eval a model")

    parser.add_argument(
        "--data_path",
        type=str,
        help="Data path",
        required=False,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model",
        required=False,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to model",
        required=False,
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=0.6,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help='Specify num of return sequences',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help='Specify max new tokens for the generation',
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help='Specify the max seq length for the context',
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help="OPT model has a fixed number (1) of padding tokens at the beginning of the input. We did not see this in other models but keep it as an option for now."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        choices=["English", "Chinese", "Japanese"]
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the outputs."
    )
    parser.add_argument(
        '--calc_rouge',
        action='store_true',
        help='Calculate rouge scores.'
    )

    args = parser.parse_args()

    return args


def generate(
    model,
    tokenizer,
    inputs,
    num_beams=1,
    num_beam_groups=1,
    do_sample=False,
    num_return_sequences=1,
    max_new_tokens=100,
    stopping_criteria=None,
    stop_words_ids=None
):

    generate_ids = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        stopping_criteria=stopping_criteria
    )

    # print(generate_ids)
    # remove stop criteria ending
    pad_token_id = tokenizer.pad_token_id
    if stop_words_ids is not None:
        for stop_id in stop_words_ids:
            for n in range(generate_ids.shape[0]):
                if torch.all((stop_id == generate_ids[n][-len(stop_id):])):
                    generate_ids[n][-len(stop_id):] = pad_token_id
                    break

    result = tokenizer.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return result, generate_ids

def print_utils(gen_output):
    for i in range(len(gen_output)):
        print()
        print(gen_output[i])
        print()


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

def cleanup_completion(completion):
    if "Human:" in completion:
        completion = completion[:completion.find("Human:")]
    if "<|endoftext|>" in completion:
        completion = completion[:completion.find("<|endoftext|>")]
    if "### Instruction:" in completion:
        completion = completion[:completion.find("### Instruction:")]
    if "### Instruction" in completion:
        completion = completion[:completion.find("### Instruction")]
    if "Comment:" in completion:
        completion = completion[:completion.find("### Instruction")]

    words = completion.split(" ")
    new_completion = []
    for word in words:
        new_completion.append(word)
        if len(new_completion) < 8: continue

        if  " ".join(new_completion[-5:]) in  " ".join(new_completion[:-5]):
            new_completion = new_completion[:-5]
            completion = " ".join(new_completion)
            break

    return completion

def get_prompt(ex_completion):
    prompt = ex_completion
    one_found = False
    for ending_phrase in ["Answer:", "Human:", "Post Summary:", "### Response:"]:
        if ending_phrase in ex_completion:
            prompt = ex_completion[:ex_completion.find(ending_phrase) + len(ending_phrase)]
            one_found = True

    if not one_found:
        print("PROMPT NOT FOUND")
    return prompt

def prompt_eval(args, model_baseline, tokenizer, device, prompts):
    results = []

    baseline_lens = []

    stop_words = ["Human:", r"\nHuman:", " Human:", "### Instruction", "### Instruction:", "Comment:", "Comment: "]
    stop_words_ids = []
    # stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze().to("cuda") for stop_word in stop_words]
    for stop_word in stop_words:
        tokenized = tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze().to("cuda")
        tokenized = tokenized[tokenized != tokenizer.pad_token_id]
        stop_words_ids.append(tokenized)
    manual_stop = torch.tensor([29950,  7889, 29901], device="cuda")  # can't figure out this issue with llama... uses strange tokens for Human:
    stop_words_ids.append(manual_stop)

    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    print("stop_words_ids")
    print(stop_words_ids)

    if True or args.calc_rouge:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        baseline_rouge_scores = []

    for prompt in prompts:
        print()
        print()
        print()
        print("==========PROMPT=========")
        print(prompt)

        # inputs = tokenizer(prompt, return_tensors="pt").to(device)
        inputs = tokenizer(
            prompt,
            max_length=args.max_seq_length,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        print("==========Baseline: Greedy=========")
        r_base, baseline_generate_ids = generate(
            model_baseline,
            tokenizer,
            inputs,
            num_beams=1,
            num_return_sequences=args.num_return_sequences,
            max_new_tokens=args.max_new_tokens,
            stopping_criteria=stopping_criteria,
            stop_words_ids=stop_words_ids
        )

        cut_prompt = tokenizer.batch_decode(
            inputs["input_ids"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        print("cut prompt", len(cut_prompt))
        print(cut_prompt)
        print()

        cleaned_up_base = cleanup_completion(r_base[0][len(cut_prompt):])
        baseline_lens.append(len(cleaned_up_base))

        baseline_rouge_score = None
        if True or args.calc_rouge:
            baseline_rouge_score = scorer.score(prompt, cleaned_up_base)
            baseline_rouge_scores.append(baseline_rouge_score)

        # print_utils(r_base)
        print("cleaned up")
        print(cleaned_up_base)

        # remove extra tokens and clean up text for easy evaluation afterwards
        results.append({
            "prompt": prompt,
            "response": cleaned_up_base,
            "rouge": baseline_rouge_score,
            "prompt_token_len": inputs.input_ids.shape[1],
            "generate_ids": baseline_generate_ids.cpu().tolist()
        })

        print("====================prompt end=============================")
        print()
        if True or args.calc_rouge:
            print("baseline_rouge_score", baseline_rouge_score)
        print()

    print()
    print()
    print()
    if True or args.calc_rouge:
        print()
        print("ROUGE SCORES")
        print()
        print("BASELINE")
        print_rouge_stats(baseline_rouge_scores)
        print()
        print()

    return results


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

def print_rouge_stats(rouge_scores):
    for key in ["precision", "recall", "fmeasure"]:
        rouge1_scores = [getattr(x["rouge1"], key) for x in rouge_scores]
        rougeL_scores = [getattr(x["rougeL"], key) for x in rouge_scores]

        print("ROUGE1", key, mean(rouge1_scores), std(rouge1_scores))
        print("ROUGEL", key, mean(rougeL_scores), std(rougeL_scores))

def main():
    args = parse_args()

    device = torch.device("cuda:0")

    args.zero_stage = 3

    print("Using zero stage", args.zero_stage)

    ds_config = get_eval_ds_config(offload=False, stage=args.zero_stage)

    if args.zero_stage == 3:
        loading_ds_config = get_eval_ds_config(offload=False, stage=0)   # necessary JUST to load in model without shard issues
    else:
        loading_ds_config = ds_config

    if args.model_path:
        model_name_or_path_baseline = args.model_path
    elif args.model_name:
        model_name_or_path_baseline = args.model_name
    else:
        print("no model to load")
        sys.exit(1)

    if os.path.exists(model_name_or_path_baseline):
        path_files = os.listdir(model_name_or_path_baseline)

        print("checking dir for ppo outputs...")
        if len(path_files) == 2 and "actor" in path_files and "critic" in path_files:
            print('found that model_name_or_path_baseline is a ppo directory, using best actor')
            print("old")
            print(model_name_or_path_baseline)
            model_name_or_path_baseline = os.path.join(model_name_or_path_baseline, "actor", "best")
            print("new")
            print(model_name_or_path_baseline)

    # tokenizer = load_hf_tokenizer(model_name_or_path_baseline, fast_tokenizer=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path_baseline,
        fast_tokenizer=True,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        truncation_side="left",
        padding_side="right"  # necessary for llama
    )
    tokenizer.pad_token = tokenizer.eos_token

    model_baseline = create_hf_model(
        AutoModelForCausalLM,
        model_name_or_path_baseline,
        tokenizer,
        None
    )

    model_baseline.to(device)
    model_baseline.eval()

    max_samples = 200
    # max_samples = 5   # temp debug

    if args.data_path is not None:
        data = load_from_disk(args.data_path)["test"]
        # prompts = list(set(data["prompt"]))[:250]  # remove duplicates...     this is not deterministic...

        # should be deterministic...?
        seen = {}
        prompts = []
        for prompt in data["prompt"]:
            if prompt not in seen:
                prompts.append(prompt)
                seen[prompt] = True
            # if len(prompts) == 250: break
            if len(prompts) == max_samples:
                break

    # One observation: if the prompt ends with a space " ", there is a high chance that
    # the original model (without finetuning) will stuck and produce no response.
    # Finetuned models have less such issue. Thus following prompts all end with ":"
    # to make it a more meaningful comparison.
    elif args.language == "English":
        # prompts = [
        #     "Human: Please tell me about Microsoft in a few sentence? Assistant:",
        #     "Human: Explain the moon landing to a 6 year old in a few sentences. Assistant:",
        #     "Human: Write a short poem about a wise frog. Assistant:",
        #     "Human: Who was president of the United States in 1955? Assistant:",
        #     "Human: How does a telescope work? Assistant:",
        #     "Human: Why do birds migrate south for the winter? Assistant:"
        # ]
        data = load_dataset("Dahoas/rm-static", split="test")


        # take only first x samples of test set to limit compute

        prompts = data["prompt"][:max_samples]

    elif args.language == "Chinese":
        prompts = [
            "Human: 请用几句话介绍一下微软? Assistant:",
            "Human: 用几句话向6岁的孩子解释登月。 Assistant:",
            "Human: 写一首关于一只聪明的青蛙的短诗。 Assistant:",
            "Human: 谁是1955年的美国总统? Assistant:", "Human: 望远镜是如何工作的? Assistant:",
            "Human: 鸟类为什么要南迁过冬? Assistant:"
        ]
    elif args.language == "Japanese":
        prompts = [
            "Human: マイクロソフトについて簡単に教えてください。 Assistant:",
            "Human: 6歳児に月面着陸を短い文で説明する。 Assistant:",
            "Human: 賢いカエルについて短い詩を書いてください。 Assistant:",
            "Human: 1955年のアメリカ合衆国大統領は誰? Assistant:",
            "Human: 望遠鏡はどのように機能しますか? Assistant:",
            "Human: 鳥が冬に南に移動するのはなぜですか? Assistant:"
        ]

    with torch.no_grad():
        results = prompt_eval(args, model_baseline, tokenizer, device, prompts)

    with open(os.path.join(args.output_dir, "outputs.jsonl"), 'w') as fp:
        for line in results:
            fp.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
