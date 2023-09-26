import os
from pathlib import Path
import datetime
import time

from azureml.core import Workspace, Dataset, Datastore
from azureml.core.authentication import InteractiveLoginAuthentication
from azure.ml.component import Component, dsl


def main():
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H%M%S')
    output_path = os.path.join('oss_rlhf', f"logs-{timestamp}")
    job_name = 'gpt-eval-stackllama'
    cpu_compute_name = 'd15'
    gpu_compute_name = 'A100-80G-PCIE-westus3'
    num_nodes = 1
    processes_per_node = 1

    ################################################
    # Connect to Azure ML workspace
    ################################################
    ws = Workspace.from_config(
        path=os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'aml_ws_configs', 'tscience_research.json'),
        auth=InteractiveLoginAuthentication(tenant_id="72f988bf-86f1-41af-91ab-2d7cd011db47")
    )
    datastore = Datastore.get(ws, 'babela100')

    ################################################
    # Load dataset(s)
    ################################################
    all_datasets_path = "misantac_oss_rlhf/stackllama_md_processed/stackllama_md_filtered_processed_150000/"
    data_eval = Dataset.File.from_files(path=[(datastore, all_datasets_path)], validate=True).as_mount()

    ################################################
    # Load base model(s)
    ################################################
    sft_model_path = Dataset.File.from_files(path=[(datastore, "hitshar_rlhf/logs-2023-09-19-132943/stackxLlama2/sft/")], validate=True).as_mount()
    rm_model_path = Dataset.File.from_files(path=[(datastore, "hitshar_rlhf/logs-2023-09-19-163558/stackxLlama/rm/2/")], validate=True).as_mount()
    ppo1_model_path = Dataset.File.from_files(path=[(datastore, "hitshar_rlhf/logs-2023-09-23-071510/stackxLlama2/ppo/actor/4999")], validate=True).as_mount()
    apa1_model_path = Dataset.File.from_files(path=[(datastore, "hitshar_rlhf/logs-2023-09-23-071833/stackxLlama2/ppo/actor/4999")], validate=True).as_mount()
    ppo2_model_path = Dataset.File.from_files(path=[(datastore, "hitshar_rlhf/logs-2023-09-24-163340/stackxLlama2/ppo/actor/3999")], validate=True).as_mount()
    apa2_model_path = Dataset.File.from_files(path=[(datastore, "hitshar_rlhf/logs-2023-09-23-071833/stackxLlama2/ppo/actor/2999")], validate=True).as_mount()

    ################################################
    # Load components
    ################################################
    components_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'components')
    model_gen_func = Component.from_yaml(ws, yaml_file=os.path.join(components_dir, 'generate_completions.yaml'))
    compare_gen_func = Component.from_yaml(ws, yaml_file=os.path.join(components_dir, 'compare_generations.yaml'))
    gpt_scorer_func = Component.from_yaml(ws, yaml_file=os.path.join(components_dir, 'gpt_scorer.yaml'))
    parse_results_func = Component.from_yaml(ws, yaml_file=os.path.join(components_dir, 'parse_results.yaml'))

    first_model_generations = None
    second_model_generations = None
    merged_generations = None
    scoring_outputs = None

    # first_model_generations = Dataset.File.from_files(path=[(datastore, "oss_rlhf/logs-2023-09-25-101314/sft-gen/")], validate=True).as_mount()
    # second_model_generations = Dataset.File.from_files(path=[(datastore, "oss_rlhf/logs-2023-09-25-101314/apa-gen/")], validate=True).as_mount()
    # merged_generations = Dataset.File.from_files(path=[(datastore, "oss_rlhf/logs-2023-09-25-101314/eval-sft-ppo/")], validate=True).as_mount()
    # scoring_outputs = Dataset.File.from_files(path=[(datastore, "oss_rlhf/logs-2023-09-25-101314/eval-sft-ppo/")], validate=True).as_mount()

    first_model_path = ppo2_model_path
    second_model_path = apa2_model_path

    first_model_name = "ppo"
    second_model_name = "apa"

    ################################################
    # Define pipeline (configure and connect components)
    ################################################
    @dsl.pipeline(
        name=f"{timestamp}-{job_name}",
        default_compute_target=gpu_compute_name,
    )
    def build_pipeline():
        if first_model_generations is None:
            gen_first_model = model_gen_func(
                model_path=first_model_path,
                num_padding_at_beginning=0,
                max_new_tokens=128,
                data_path=data_eval,
            )
            gen_first_model.runsettings.resource_layout.configure(instance_count=num_nodes, process_count_per_node=processes_per_node)
            gen_first_model.outputs.output_dir.configure(
                mode="mount",
                path_on_datastore=os.path.join(output_path, f"{first_model_name}-gen"),
                datastore=datastore
            )
            first_model_outputs = gen_first_model.outputs.output_dir
        else:
            first_model_outputs = first_model_generations

        if second_model_generations is None:
            gen_second_model = model_gen_func(
                model_path=second_model_path,
                num_padding_at_beginning=0,
                max_new_tokens=128,
                data_path=data_eval,
            )
            gen_second_model.runsettings.resource_layout.configure(instance_count=num_nodes, process_count_per_node=processes_per_node)
            gen_second_model.outputs.output_dir.configure(
                mode="mount",
                path_on_datastore=os.path.join(output_path, f"{second_model_name}-gen"),
                datastore=datastore
            )
            second_model_outputs = gen_second_model.outputs.output_dir
        else:
            second_model_outputs = second_model_generations

        if merged_generations is None:
            merge_first_vs_second = compare_gen_func(
                baseline_model_outputs=first_model_outputs,
                finetune_model_outputs=second_model_outputs,
                num_padding_at_beginning=0,
                reward_model_path=rm_model_path,
            )
            merge_first_vs_second.runsettings.resource_layout.configure(instance_count=num_nodes, process_count_per_node=processes_per_node)
            merge_first_vs_second.outputs.output_dir.configure(
                mode="mount",
                path_on_datastore=os.path.join(output_path, f"eval-{first_model_name}-{second_model_name}"),
                datastore=datastore
            )
            compare_outputs = merge_first_vs_second.outputs.output_dir
        else:
            compare_outputs = merged_generations

        if scoring_outputs is None:
            gpt_scorer = gpt_scorer_func(
                data_dir=compare_outputs,
            )
            gpt_scorer.compute = cpu_compute_name
            gpt_scorer.outputs.output_dir.configure(
                mode="mount",
                path_on_datastore=os.path.join(output_path, f"eval-{first_model_name}-{second_model_name}"),
                datastore=datastore
            )
            gpt_scorer.runsettings.environment_variables = {
                'OPENAI_API_BASE': os.getenv("OPENAI_API_BASE"),
                'OPENAI_API_VERSION': os.getenv("OPENAI_API_VERSION"),
                'OPENAI_API_KEY': os.getenv("OPENAI_API_KEY")
            }
            gpt_scorer_outputs = gpt_scorer.outputs.output_dir
        else:
            gpt_scorer_outputs = scoring_outputs

        parse_results = parse_results_func(
            data_dir=gpt_scorer_outputs,
            baseline_evaluations="gpt4_outputs_baseline.jsonl",
            finetuned_evaluations="gpt4_outputs_finetuned.jsonl",
        )
        parse_results.compute = cpu_compute_name
        parse_results.outputs.output_dir.configure(
            mode="mount",
            path_on_datastore=os.path.join(output_path, f"eval-{first_model_name}-{second_model_name}"),
            datastore=datastore
        )

    tags = {
        "dataset": "stack-exchange",
    }

    pipeline = build_pipeline()
    run = pipeline.submit(
        tags=tags,
        experiment_name="deepspeed-chat-14b84db0",
        regenerate_outputs=False,
    )


if __name__ == "__main__":
    main()
