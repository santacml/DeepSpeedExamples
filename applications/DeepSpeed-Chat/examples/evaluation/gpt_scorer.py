import argparse
import json
import logging
import os

import openai
from retry import retry


def load_generations(file):
    generations = []
    num_generations = 0
    with open(file, "r") as fp:
        for line in fp:
            if any(not x.isspace() for x in line):
                generations.append(json.loads(line))
                num_generations += 1

    print("num results", num_generations)
    return generations

def load_json(filepath):
    with open(filepath) as f:
        return json.load(f)

def dump_json(filepath, content, indent=4):
    if isinstance(content, str):
        content = json.loads(content)

    with open(filepath, "w") as f:
        json.dump(content, f, indent=indent)


class ChatEngine(object):
    def __init__(self, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, stop, sleep=1):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.sleep = sleep

    def get_response(self, system_msg, input):
        raise NotImplementedError

    def _get_conversation_template(self):
        return {
            "conversation_id": "",
            "model": self.model,
            "version": openai.api_version,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "max_tokens": self.max_tokens,
            "stop": self.stop,
        }


class OpenAIEngine(ChatEngine):
    def __init__(
        self,
        api_type,
        api_base,
        api_version,
        api_key,
        generation_type='chat',
        model='gpt-4-32k',
        temperature=0,
        max_tokens=2048,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        sleep=1
    ):
        openai.api_type = api_type
        openai.api_base = api_base
        openai.api_version = api_version
        openai.api_key = api_key

        self.generation_type = generation_type
        super().__init__(model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, stop, sleep)

    @retry(tries=10, delay=30, backoff=2, max_delay=90, logger=logging.getLogger(__name__))
    def get_response(self, user_msg, system_msg=None):
        messages = []
        if system_msg:
            messages.append(system_msg)

        messages.append(user_msg)
        response = openai.ChatCompletion.create(
            engine=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=self.stop,
            messages=messages
        )

        return response["choices"][0]["message"]["content"]


def main(args):
    engine = OpenAIEngine(
        api_type="azure",
        api_base=os.getenv("OPENAI_API_BASE"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model='gpt-4-32k',
        generation_type="chat"
    )

    templates_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'prompt_template.json')
    templates = load_json(templates_path)
    system_msg_template = templates['SYSTEM_MSG']
    user_msg_template = templates['USER_MSG']

    generations_path = os.path.join(args.data_dir, 'outputs.jsonl')
    generations = load_generations(generations_path)

    os.makedirs(args.output_dir, exist_ok=True)
    for model in ['baseline', 'finetuned']:
        gpt4_outputs = []
        for i, generation in enumerate(generations):
            user_msg = user_msg_template.copy()
            user_msg['content'] = user_msg['content'].replace('{prompt}', generation['prompt']).replace('{completion}', generation[model]).replace('{ground_truth}', generation['ground_truth'])

            samples = []
            for _ in range(3):
                response = engine.get_response(user_msg, system_msg_template)
                samples.append(response)

            gpt4_outputs.append({
                'metadata': generation,
                'samples': samples,
            })

        if i % 10 == 9:
            with open(os.path.join(args.output_dir, f"gpt4_outputs_{model}.jsonl"), 'a') as fp:
                for line in gpt4_outputs:
                    fp.write(json.dumps(line) + "\n")
            print('saved', i, 'outputs')
            gpt4_outputs = []

    if gpt4_outputs:
        with open(os.path.join(args.output_dir, f"gpt4_outputs_{model}.jsonl"), 'a') as fp:
            for line in gpt4_outputs:
                fp.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="")
    parser.add_argument("--output_dir", help="")
    args = parser.parse_args()
    main(args)
