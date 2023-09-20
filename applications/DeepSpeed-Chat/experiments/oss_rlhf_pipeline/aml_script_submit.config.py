import os
from azureml.core import (
    Dataset,
    Datastore,
    Experiment,
    ScriptRunConfig,
    Workspace,
)

from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.runconfig import PyTorchConfiguration
from azureml.data import OutputFileDatasetConfig
from azureml.core.compute import ComputeTarget
from azureml.core import Environment

from hydra_rlhf.azureml_utils import compute_utils, environment_utils

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir)

ws = Workspace.from_config(
    path=os.path.join(root_dir, 'aml_ws_configs', 'tscience_research.json'),
    auth=InteractiveLoginAuthentication(tenant_id="72f988bf-86f1-41af-91ab-2d7cd011db47")
)

default_ds = ws.get_default_datastore()
babela100_ds = Datastore.get(ws, 'babela100')

installation_cmds = "pip install -e . && "

sft_run_config = ScriptRunConfig(
    source_directory=os.path.join(root_dir),
    command=[
        installation_cmds + "python", "./examples/training/step1_supervised_finetuning/main.py",
        "--data_path", Dataset.File.from_files(babela100_ds.path("misantac_oss_rlhf/stackllama_md_processed/stackllama_md_filtered_processed_150000/**")).as_mount(),
        "--data_split", "4,2,4",
        "--model_dir", Dataset.File.from_files(path=[(babela100_ds, "llama-2")], validate=True).as_mount(),
        "--model_name_or_path", "Llama-2-7b-hf",
        "--per_device_train_batch_size", "1",
        "--per_device_eval_batch_size", "2",
        "--max_seq_len", "800",
        "--learning_rate", "5e-7",
        "--weight_decay", ".1",
        "--num_train_epochs", "5",
        "--gradient_accumulation_steps", "3",
        "--lr_scheduler_type", "cosine",
        "--num_warmup_steps", "0",
        "--seed", "42",
        "--zero_stage", "1",
        "--lora_dim", "0",
        "--lora_module_name", "layers",
        "--deepspeed",
        "--only_optimize_lora",
        "--output_dir", OutputFileDatasetConfig(destination=(default_ds, "rlhf_models/sft"))
    ],
    compute_target=ComputeTarget(workspace=ws, name="A100-80G-PCIE-westus3"),
    environment=Environment.get(workspace=ws, name="hydra-rlhf-env"),
    distributed_job_config=PyTorchConfiguration(process_count=4, node_count=1),
)

model_dir_dataset = Dataset.File.from_files(path=[(babela100_ds, "oss_rlhf/logs-2023-09-11-211800")], validate=True).as_mount()
rl_run_config = ScriptRunConfig(
    source_directory=os.path.join(root_dir),
    command=[
        installation_cmds + "python", "./examples/training/step3_rlhf_finetuning/main.py",
        "--data_path", Dataset.File.from_files(babela100_ds.path("misantac_oss_rlhf/stackllama_md_processed/stackllama_md_filtered_processed_150000/**")).as_mount(),
        "--data_split", "0,0,1",
        "--actor_model_dir", model_dir_dataset,
        "--actor_model_name_or_path", "sft",
        "--critic_model_dir", model_dir_dataset,
        "--critic_model_name_or_path", "rm",
        "--num_padding_at_beginning", "0",
        "--per_device_training_batch_size", "2",
        "--per_device_generation_batch_size", "2",
        "--generation_batches", "2",
        "--ppo_epochs", "4",
        "--max_answer_seq_len", "400",
        "--max_prompt_seq_len", "400",
        "--actor_learning_rate", "5e-5",
        "--critic_learning_rate", "5e-5",
        "--actor_weight_decay", "0",
        "--critic_weight_decay", "0",
        "--num_train_epochs", "1",
        "--gradient_accumulation_steps", "25",
        "--lr_scheduler_type", "cosine",
        "--num_warmup_steps", "100",
        "--seed", "42",
        "--actor_zero_stage", "1",
        "--critic_zero_stage", "1",
        "--actor_lora_dim", "128",
        "--actor_lora_module_name", "layers",
        "--critic_lora_dim", "128",
        "--critic_lora_module_name", "layers",
        "--normalize_rm_scale", "1.0",
        "--disable_actor_dropout",
        "--disable_critic_dropout",
        "--only_optimize_lora",
        "--deepspeed",
        "--output_dir", OutputFileDatasetConfig(destination=(default_ds, "rlhf_models/rl"))
    ],
    compute_target=ComputeTarget(workspace=ws, name="A100-80G-PCIE-westus3"),
    environment=Environment.get(workspace=ws, name="hydra-rlhf-env"),
    distributed_job_config=PyTorchConfiguration(process_count=4, node_count=1),
)

exp = Experiment(workspace=ws, name="deepspeed-chat-14b84db0")
exp.submit(config=rl_run_config)
