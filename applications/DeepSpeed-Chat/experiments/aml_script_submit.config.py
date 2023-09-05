import os
from azureml.core import (
    Environment,
    Datastore,
    Experiment,
    ScriptRunConfig,
    Workspace,
)

from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.runconfig import PyTorchConfiguration
from azureml.data import OutputFileDatasetConfig

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)

ws = Workspace.from_config(
    path=os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, 'aml_ws_configs', 'tscience_research.json'),
    auth=InteractiveLoginAuthentication(tenant_id="72f988bf-86f1-41af-91ab-2d7cd011db47")
)

default_ds = ws.get_default_datastore()
babela100_ds = Datastore.get(ws, 'babela100')

installation_cmds = "pip install -e .[all] && "

script_run_config = ScriptRunConfig(
    source_directory=os.path.join(root_dir),
    command=[
        installation_cmds + "deepspeed ./examples/training/step1_supervised_finetuning/main.py",
        "--data_path", "Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets",
        "--data_split", "2,4,4",
        "--model_name_or_path", "meta-llama/Llama-2-7b-hf",
        "--per_device_train_batch_size", "4",
        "--per_device_eval_batch_size", "4",
        "--max_seq_len", "512",
        "--learning_rate", "9.65e-6",
        "--weight_decay", "0",
        "--num_train_epochs", "4",
        "--gradient_accumulation_steps", "1",
        "--lr_scheduler_type", "cosine",
        "--num_warmup_steps", "0",
        "--seed", "1234",
        "--gradient_checkpointing",
        "--zero_stage", "3",
        "--deepspeed",
        "--lora_dim", "128",
        "--lora_module_name", "layers.",
        "--output_dir", OutputFileDatasetConfig(destination=(default_ds, "deepspeed-chat/sft"))
    ],
    compute_target=ws.compute_targets["A100-80G-PCIE-westus3"],
    environment=Environment.get(workspace=ws, name="AzureML-pytorch-1.9-ubuntu18.04-py37-cuda11-gpu").clone("deepspeed-chat-env"),
    # environment=Environment.from_dockerfile(
    #     "deepspeed-chat-env",
    #     os.path.join(root_dir, "experiments", "dockerfile")
    # ),
    distributed_job_config=PyTorchConfiguration(process_count=8, node_count=1),
)

exp = Experiment(workspace=ws, name="deepspeed-chat-rlhf")
exp.submit(config=script_run_config)
