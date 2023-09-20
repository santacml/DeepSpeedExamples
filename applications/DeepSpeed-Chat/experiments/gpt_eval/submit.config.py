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
    job_name = 'rlhf-stack-exchange-llama2'
    compute_name = 'A100-80G-PCIE-westus3'
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
    sft_model_path = Dataset.File.from_files(path=[(datastore, "misantac_oss_rlhf/SAVED_FINAL_RUNS/logs-2023-08-02-171401/sft")], validate=True).as_mount()
    rm_model_path = Dataset.File.from_files(path=[(datastore, "misantac_oss_rlhf/SAVED_FINAL_RUNS/logs-2023-07-11-104249/rm")], validate=True).as_mount()
    rl_model_path = Dataset.File.from_files(path=[(datastore, "misantac_oss_rlhf/SAVED_FINAL_RUNS/logs-2023-08-02-171401/ppo/actor/best")], validate=True).as_mount()

    ################################################
    # Load components
    ################################################
    components_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'components')
    model_gen_func = Component.from_yaml(ws, yaml_file=os.path.join(components_dir, 'model_generation.yaml'))
    compare_gen_func = Component.from_yaml(ws, yaml_file=os.path.join(components_dir, 'compare_generations.yaml'))

    ################################################
    # Define pipeline (configure and connect components)
    ################################################
    @dsl.pipeline(
        name=f"{timestamp}-{job_name}",
        default_compute_target=compute_name,
    )
    def build_pipeline():
        eval_sft = model_gen_func(
            model_path=sft_model_path,
            num_padding_at_beginning=0,
            max_new_tokens=512,
            data_path=data_eval,
        )
        eval_sft.runsettings.resource_layout.configure(instance_count=num_nodes, process_count_per_node=processes_per_node)
        eval_sft.outputs.output_dir.configure(
            mode="mount",
            path_on_datastore=os.path.join(output_path, "sft-gen"),
            datastore=datastore
        )

        eval_rl = model_gen_func(
            model_path=rl_model_path,
            num_padding_at_beginning=0,
            max_new_tokens=512,
            data_path=data_eval,
        )
        eval_rl.runsettings.resource_layout.configure(instance_count=num_nodes, process_count_per_node=processes_per_node)
        eval_rl.outputs.output_dir.configure(
            mode="mount",
            path_on_datastore=os.path.join(output_path, "rl-gen"),
            datastore=datastore
        )

        eval_sft_vs_rl = compare_gen_func(
            baseline_model_outputs=eval_sft.outputs.output_dir,
            finetune_model_outputs=eval_rl.outputs.output_dir,
            num_padding_at_beginning=0,
            reward_model_path=rm_model_path,
        )
        eval_sft_vs_rl.runsettings.resource_layout.configure(instance_count=num_nodes, process_count_per_node=processes_per_node)
        eval_sft_vs_rl.outputs.output_dir.configure(
            mode="mount",
            path_on_datastore=os.path.join(output_path, "eval-sft-rl"),
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
