import os
from pathlib import Path
import datetime
import time

from hydra.utils import get_original_cwd, to_absolute_path
import hydra
from azureml.core import Workspace, Dataset, Datastore
from azure.ml.component import Component, dsl


@hydra.main(config_path="configs", config_name="rlhf_se_llama2")
def main(config):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H%M%S')
    output_path = os.path.join(config.outputs.output_dir, f"logs-{timestamp}")

    ################################################
    # Connect to Azure ML workspace
    ################################################
    ws = Workspace(
        subscription_id=config.aml_config.subscription_id,
        resource_group=config.aml_config.resource_group,
        workspace_name=config.aml_config.workspace_name,
    )
    datastore = Datastore.get(ws, config.aml_config.datastore)

    ################################################
    # Load dataset(s)
    ################################################
    data_sft_train = Dataset.File.from_files(path=[(datastore, config.datasets.all.path)], validate=True).as_mount()
    data_sft_val = Dataset.File.from_files(path=[(datastore, config.datasets.all.path)], validate=True).as_mount()
    data_rm_train = Dataset.File.from_files(path=[(datastore, config.datasets.all.path)], validate=True).as_mount()
    data_rm_val = Dataset.File.from_files(path=[(datastore, config.datasets.all.path)], validate=True).as_mount()
    data_ppo_train = Dataset.File.from_files(path=[(datastore, config.datasets.all.path)], validate=True).as_mount()
    data_eval = Dataset.File.from_files(path=[(datastore, config.datasets.all.path)], validate=True).as_mount()

    ################################################
    # Load base model(s)
    ################################################
    model_dir = Dataset.File.from_files(path=[(datastore, config.models.base_model_path)], validate=True).as_mount()

    ################################################
    # Load components
    ################################################
    sft_train_func = Component.from_yaml(ws, yaml_file=to_absolute_path(f"{config.components.sft}"))
    rm_train_func = Component.from_yaml(ws, yaml_file=to_absolute_path(f"{config.components.rm}"))
    rl_func = Component.from_yaml(ws, yaml_file=to_absolute_path(f"{config.components.ppo}"))
    single_eval_func = Component.from_yaml(ws, yaml_file=to_absolute_path(f"{config.components.single_eval}"))
    compare_eval_func = Component.from_yaml(ws, yaml_file=to_absolute_path(f"{config.components.compare_eval}"))

    sft_model_input_path = None
    rm_model_input_path = None
    ppo_model_input_path = None

    ################################################
    # Define pipeline (configure and connect components)
    ################################################
    @dsl.pipeline(
        name=f"{timestamp}-{config.run.job_name}-{config.models.model_size}",
        default_compute_target=config.aml_config.gpu_target,
    )
    def build_pipeline():
        ################################################
        # SFT (LM): fine-tune from base LM model
        ################################################
        if sft_model_input_path is None:
            sft_trainer = sft_train_func(
                data_names=None,
                data_path=data_sft_train,
                data_split="4,2,4",
                model_dir=model_dir,
                model_name_or_path="",
                per_device_train_batch_size=1,
                per_device_eval_batch_size=2,
                max_seq_len=400 + 400,
                learning_rate=5e-7,
                weight_decay=.1,
                num_train_epochs=5,
                gradient_accumulation_steps=3,
                lr_scheduler_type="cosine",
                num_warmup_steps=0,
                seed=42,
                zero_stage="1",
                lora_dim=0,
                lora_module_name="layers",
            )
            sft_trainer.runsettings.resource_layout.configure(instance_count=1, process_count_per_node=8)
            sft_trainer.outputs.output_dir.configure(
                mode="mount",
                path_on_datastore=os.path.join(output_path, "sft"),
                datastore=datastore
            )
            sft_model_path = sft_trainer.outputs.output_dir
        else:
            sft_model_path = sft_model_input_path

        ################################################
        # RM: fine-tune from SFT RM
        ################################################
        if rm_model_input_path is None:
            rm_trainer = rm_train_func(
                data_names=None,
                data_path=data_rm_train,
                data_split="2,4,4",
                model_name_or_path="",
                model_dir=model_dir,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=2,
                max_seq_len=400 + 400,
                num_padding_at_beginning=0,
                learning_rate=1e-6,
                weight_decay=.1,
                num_train_epochs=3,
                gradient_accumulation_steps=12,
                lr_scheduler_type="cosine",
                num_warmup_steps=0,
                seed=42,
                zero_stage="1",
                lora_dim=0,
                lora_module_name="layers",
            )
            rm_trainer.runsettings.resource_layout.configure(instance_count=1, process_count_per_node=8)
            rm_trainer.outputs.output_dir.configure(
                mode="mount",
                path_on_datastore=os.path.join(output_path, "rm"),
                datastore=datastore
            )
            rm_model_path = rm_trainer.outputs.output_dir
        else:
            rm_model_path = rm_model_input_path

        ################################################
        # PPO: fine-tune from SFT LM
        ################################################
        if ppo_model_input_path is None:
            ppo_trainer = rl_func(
                data_names=None,
                data_path=data_ppo_train,
                data_split="2,4,4",
                actor_model_dir=sft_model_path,
                actor_model_name_or_path=None,
                critic_model_dir=rm_model_path,
                critic_model_name_or_path=None,
                num_padding_at_beginning=0,
                per_device_generation_batch_size=1,
                per_device_training_batch_size=3,
                generation_batches=1,
                ppo_epochs=1,
                max_answer_seq_len=400,
                max_prompt_seq_len=400,
                actor_learning_rate=5e-4,
                critic_learning_rate=5e-5,
                actor_weight_decay=0,
                critic_weight_decay=0,
                normalize_rm_scale=1.0,
                num_train_epochs=1,
                lr_scheduler_type="cosine",
                gradient_accumulation_steps=25,
                num_warmup_steps=100,
                seed=42,
                actor_zero_stage="1",
                critic_zero_stage="1",
                actor_lora_dim=128,
                actor_lora_module_name="layers",
                critic_lora_dim=128,
                critic_lora_module_name="layers",
                multilora_mode="none",
                save_strategy="reward"
            )
            ppo_trainer.runsettings.resource_layout.configure(instance_count=1, process_count_per_node=8)
            ppo_trainer.outputs.output_dir.configure(
                mode="mount",
                path_on_datastore=os.path.join(output_path, "ppo"),
                datastore=datastore
            )
            ppo_model_path = ppo_trainer.outputs.output_dir
        else:
            ppo_model_path = ppo_model_input_path

        ################################################
        # Batch Inference Eval
        ################################################
        eval_sft = single_eval_func(
            model_path_baseline=sft_model_path,
            num_padding_at_beginning=0,
            max_new_tokens=400,
            data_path=data_eval,
        )
        eval_sft.runsettings.resource_layout.configure(instance_count=1, process_count_per_node=1)

        eval_ppo = single_eval_func(
            model_path_baseline=ppo_model_path,
            num_padding_at_beginning=0,
            max_new_tokens=400,
            data_path=data_eval,
        )
        eval_ppo.runsettings.resource_layout.configure(instance_count=1, process_count_per_node=1)

        eval_sft_vs_ppo = compare_eval_func(
            baseline_model_outputs=eval_sft.outputs.output_dir,
            finetune_model_outputs=eval_ppo.outputs.output_dir,
            num_padding_at_beginning=0,
            reward_model_path=rm_model_path,
        )
        eval_sft_vs_ppo.runsettings.resource_layout.configure(instance_count=1, process_count_per_node=1)

    tags = {
        "dataset": config.datasets.display_name,
    }
    tags.update(config.run.tags)

    pipeline = build_pipeline()
    run = pipeline.submit(
        tags=tags,
        experiment_name=config.run.experiment_name,
        regenerate_outputs=config.run.regenerate_outputs,
    )


if __name__ == "__main__":
    main()
