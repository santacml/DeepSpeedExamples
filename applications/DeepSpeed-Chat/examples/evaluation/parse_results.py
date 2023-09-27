import argparse
import json
import os


def load_results(file):
    results = []
    num_results = 0
    with open(file, "r") as fp:
        for line in fp:
            if any(not x.isspace() for x in line):
                results.append(json.loads(line))
                # break
                num_results += 1

    print("num results", num_results)
    return results

def isdigit(c):
    return c in "0123456789"

def process_scores(samples):
    scores = []

    counts = {}
    for sample in samples:

        substring = "The integer score out of 5 for this answer is: " 

        indexes = [sample.find(substring)]
        index = indexes[0]
        while index != -1:
            index = sample.find(substring, index + 1)

            if index != -1:
                indexes.append(index)

        if len(indexes) > 1 or len(indexes) == 0:
            continue
        else:
            loc = indexes[0] + len(substring)
            if loc < len(sample) and isdigit(sample[loc]):
                score = int(sample[loc])
            else:
                print()
                print("error", loc, len(sample))
                print("sample", sample)
                print()
                continue

        scores.append(score)
    return mean(scores)

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

def main(args):
    baseline_evaluations = load_results(check_dir_get_file(os.path.join(args.data_dir, args.baseline_evaluations)))

    finetuned_evaluations = load_results(check_dir_get_file(os.path.join(args.data_dir, args.finetuned_evaluations)))

    baseline_prompts = []
    baseline_prompt_to_scores = {}
    for result in baseline_evaluations:
        if "samples" in result.keys():
            samples = result["samples"]
        elif "choices" in result.keys() :
            samples = [choice["text"] for choice in result["choices"]]
        else:
            print("no valid keys")
            print(result.keys())
            continue

        baseline_prompt_to_scores[result["metadata"]["prompt"]] = process_scores(samples)
        baseline_prompts.append(result["metadata"]["prompt"])
    print("baseline_prompts", len(baseline_prompts))

    finetuned_prompts = []
    finetuned_1_prompt_to_scores = {}
    for result in finetuned_evaluations:
        if "samples" in result.keys():
            samples = result["samples"]
        elif "choices" in result.keys():
            samples = [choice["text"] for choice in result["choices"]]
        else:
            print("no valid keys")
            print(result.keys())
            continue

        finetuned_1_prompt_to_scores[result["metadata"]["prompt"]] = process_scores(samples)
        finetuned_prompts.append(result["metadata"]["prompt"])
    print("finetuned_prompts", len(finetuned_prompts))

    intersect_prompts = [x for x in finetuned_prompts if x in baseline_prompts]

    print("intersect_prompts", len(intersect_prompts))

    total_finetune_wins = 0
    total_baseline_wins = 0
    total_ties = 0
    total_prompts = 0

    baseline_scores = []
    finetune_scores = []

    total_not_found = 0
    mismatching = 0
    for prompt in intersect_prompts:
        baseline_score = baseline_prompt_to_scores[prompt]
        finetuned_score = finetuned_1_prompt_to_scores[prompt]

        finetune_wins = 0
        baseline_wins = 0
        ties = 0

        print("eval prompt")
        print("baseline score", baseline_score, "finetune score", finetuned_score)

        if baseline_score != -1:
            baseline_scores.append(baseline_score)
        if finetuned_score != -1:
            finetune_scores.append(finetuned_score)

        if baseline_score == -1 or finetuned_score == -1:
            total_not_found += 1
        elif baseline_score > finetuned_score:
            baseline_wins += 1

        elif finetuned_score > baseline_score:
            finetune_wins += 1
        else:
            ties += 1

        if finetune_wins > baseline_wins and finetune_wins > ties:
            total_finetune_wins += 1

        elif baseline_wins > finetune_wins and baseline_wins > ties:
            total_baseline_wins += 1

        else:
            total_ties += 1

        total_prompts += 1

    print()
    print(f"Total baseline wins: {total_baseline_wins}, {total_baseline_wins/total_prompts} of total")
    print(f"Total finetune wins: {total_finetune_wins}, {total_finetune_wins/total_prompts} of total")
    print(f"Total ties: {total_ties}, {total_ties/total_prompts} of total")
    print("mismatching decisions %", mismatching / total_prompts)
    print("not found decisions %", total_not_found / total_prompts)
    print("Mean baseline score", mean(baseline_scores), "std", std(baseline_scores))
    print("Mean finetune score", mean(finetune_scores), "std", std(finetune_scores))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="")
    parser.add_argument("--baseline_evaluations", help="")
    parser.add_argument("--finetuned_evaluations", help="")
    parser.add_argument("--output_dir", help="")
    args = parser.parse_args()
    main(args)
