from pathlib import Path
import datetime
import time

from azureml.core import Workspace
from azure.ml.component import (
    Component,
    dsl,
)

from azureml.core import Workspace, Dataset, Datastore
from azureml.data import OutputFileDatasetConfig
import itertools
from azureml.core.authentication import InteractiveLoginAuthentication

# https://learn.microsoft.com/en-us/azure/machine-learning/how-to-train-distributed-gpu?view=azureml-api-2
# https://github.com/Lightning-AI/lightning/issues/13567
# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
# https://nlp.stanford.edu/mistral/tutorials/deepspeed.html
# https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training

# bug in ds
# https://github.com/microsoft/DeepSpeed/issues/3454
# https://github.com/microsoft/DeepSpeed/blob/d10b8ca011b18eba3a6ca56f4208a732d7fbb744/deepspeed/runtime/hybrid_engine.py#LL321C16-L321C37

# ds integration guide hf
# https://huggingface.co/docs/transformers/main_classes/deepspeed



def get_info(cluster):

    if cluster == "tscience":
        subscription_id = '9ec1d932-0f3f-486c-acc6-e7d78b358f9b'
        resource_group = 'tscience'
        workspace_name = 'tscience_research'
        # default_compute_target = "A100-80-WUS3"
        default_compute_target = "tscience-a100-80g-eastus"
        # default_compute_target = "a100-40g-westus3"
        ws = Workspace(subscription_id, resource_group, workspace_name)
        ds = Datastore.get(ws, "babela100")
        process_count_per_node=8
    elif cluster == "tscience-2":
        subscription_id = '9ec1d932-0f3f-486c-acc6-e7d78b358f9b'
        resource_group = 'tscience'
        workspace_name = 'tscience_research'
        default_compute_target = "A100-80G-PCIE-westus3"
        ws = Workspace(subscription_id, resource_group, workspace_name)
        ds = Datastore.get(ws, "babela100")
        process_count_per_node=4
    else:
        subscription_id = '79f57c16-00fe-48da-87d4-5192e86cd047'
        resource_group = 'alexander256'
        workspace_name = 'Alexander256V100'
        default_compute_target = "V10032G"
        ws = Workspace(subscription_id, resource_group, workspace_name)
        ds = Datastore.get(ws, "babela100")
        process_count_per_node=8

    return default_compute_target, ws, ds, process_count_per_node

def main():
    # model = "opt_1.3b"
    # model = "llama_2_7b"
    # model = "sep_rm"
    # model = "pythia_reward"

    model = "phi15"

    # TASK = "Dahoas/rm-static"
    # TASK = "sep_rm"

    TASK = "gpt4llm"

    
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H%M%S')
    output_path = f'misantac_oss_rlhf/sep_rm/logs-{timestamp}'

    ppo_baseline_input = None
    rm_model_weights_input = None
    sft_model_weights_input = None

    ## global params
    sft_model_name = None
    rm_model_name = None
    num_padding_at_beginning=0
    sft_wd = .1
    rm_wd = .1
    ppo_actor_wd = 0
    ppo_critic_wd = 0

    # lora_module_name = "self_attn"
    lora_module_name = "layers"

    zero_stage = "1"
    ppo_zero_stage = "1"
    ppo_instance_count = instance_count = 1
    ppo_baseline_input = None
    rm_model_weights_input = None
    sft_model_weights_input = None
    dpo_model_weights_input = None

    
    lora_sft_model_weights_input = None

    data_split="2,4,4"
    
    # data_split="3,3,4"

    model_name_or_path = None
    model_dir = None

    reward_model_dir = None
    reward_model_name_or_path = None


    if TASK == "sep_rm":
        default_compute_target, ws, ds, process_count_per_node = get_info("tscience")
        
        # default_compute_target, ws, ds, process_count_per_node = get_info("tscience-2")
        # instance_count = 2

        all_datasets_path =  Dataset.File.from_files(path=[(ds, "misantac_oss_rlhf/sep_rm/alpaca_processed")],validate=True).as_mount()


        per_device_train_batch_size = 2
        per_device_eval_batch_size = 4
        
        gradient_accumulation_steps=1
        
        sft_lora = 0
        sft_e = 5
        sft_lr = 1e-6

        # rm_e = 5
        rm_e = 2
        rm_lr = 1e-6
        rm_lora = 0
        
        reward_loss_mult=.1


        ppo_per_device_train_batch_size = 1
        per_device_mini_train_batch_size= 2 
        # ppo_gradient_accumulation_steps=20
        ppo_gradient_accumulation_steps=30
    
    
        
        actor_lora_learning_rate=5e-4
        critic_lora_learning_rate=5e-4
        
        max_prompt_seq_len = 256
        max_answer_seq_len = 256

        # '''
        

        if model == "opt_1.3b":
            model_weights = None
            num_padding_at_beginning = 0
            
            model_dir = None
            model_name_or_path = "facebook/opt-1.3b"
            
        elif model == "llama_2_7b":
            model_weights = None
            num_padding_at_beginning = 0

            model_dir = Dataset.File.from_files(path=[(ds, "llama-2/Llama-2-7b-hf/")],validate=True).as_mount()
            # model_dir = Dataset.File.from_files(path=[(ds, "llama-2/")],validate=True).as_mount()
            # model_name_or_path = "Llama-2-7b-hf"
            model_name_or_path = ""

        elif model == "sep_rm":
            model_weights = None
            num_padding_at_beginning = 0
            sft_model_weights_input = Dataset.File.from_files(path=[(ds, "misantac_oss_rlhf/sep_rm/logs-2023-09-10-121323/sft")],validate=True).as_mount()
            rm_model_weights_input = Dataset.File.from_files(path=[(ds, "misantac_oss_rlhf/sep_rm/rlhf-reward-v10-DistillGPT4PreferencesV2-bs64-llama2-7b-noMath-combinedAxioms0-5/checkpoint-16000_automodel/")],validate=True).as_mount()
            
            model_dir = Dataset.File.from_files(path=[(ds, "llama-2/Llama-2-7b-hf/")],validate=True).as_mount()

        elif model == "pythia_reward":

            
            model_weights = None
            num_padding_at_beginning = 0
            sft_model_weights_input = Dataset.File.from_files(path=[(ds, "misantac_oss_rlhf/sep_rm/logs-2023-09-10-121323/sft")],validate=True).as_mount()
            # rm_model_weights_input = Dataset.File.from_files(path=[(ds, "misantac_oss_rlhf/sep_rm/rlhf-reward-v10-DistillGPT4PreferencesV2-bs64-llama2-7b-noMath-combinedAxioms0-5/checkpoint-16000_automodel/")],validate=True).as_mount()

            rm_model_weights_input = None

            # rm_model_weights_input = Dataset.File.from_files(path=[(ds, "misantac_oss_rlhf/logs-2023-06-20-165325/rm")],validate=True).as_mount()

            reward_model_name_or_path =  "OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1"
            
            model_dir = Dataset.File.from_files(path=[(ds, "llama-2/Llama-2-7b-hf/")],validate=True).as_mount()


        else:
            0/0


    elif TASK == "gpt4llm":
        default_compute_target, ws, ds, process_count_per_node = get_info("alexander")
        
        # default_compute_target, ws, ds, process_count_per_node = get_info("tscience-2")
        # instance_count = 2

        all_datasets_path =  Dataset.File.from_files(path=[(ds, "misantac_oss_rlhf/instruct_gpt4_labeled_data/")],validate=True).as_mount()


        per_device_train_batch_size = 3
        per_device_eval_batch_size = 4
        
        gradient_accumulation_steps=1
        
        sft_lora = 0
        sft_e = 5
        sft_lr = 1e-6

        # rm_e = 5
        rm_e = 5
        rm_lr = 1e-6
        rm_lora = 0
        
        reward_loss_mult=.1


        ppo_per_device_train_batch_size = 1
        per_device_mini_train_batch_size= 2
        # ppo_gradient_accumulation_steps=20
        ppo_gradient_accumulation_steps=30
    
    
        
        actor_lora_learning_rate=5e-4
        critic_lora_learning_rate=5e-4
        
        max_prompt_seq_len = 256
        max_answer_seq_len = 256

        # '''
        

        if model == "phi15":

            
            model_weights = None
            model_name_or_path = "microsoft/phi-1_5"
            num_padding_at_beginning = 0

            sft_model_weights_input = Dataset.File.from_files(path=[(ds, "misantac_oss_rlhf/sep_rm/logs-2023-09-19-160343/sft")],validate=True).as_mount()


            rm_model_weights_input = None

            rm_model_weights_input = Dataset.File.from_files(path=[(ds, "misantac_oss_rlhf/sep_rm/logs-2023-09-19-160343/rm")],validate=True).as_mount()

            # reward_model_name_or_path =  "OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1"
            
            # model_dir = Dataset.File.from_files(path=[(ds, "llama-2/Llama-2-7b-hf/")],validate=True).as_mount()


        else:
            0/0

      
        
    ppo_instance_count = instance_count

    rm_gradient_accumulation_steps = int(gradient_accumulation_steps*2)

        
    max_seq_len = max_prompt_seq_len + max_answer_seq_len

    train_func = Component.from_yaml(
        ws,
        yaml_file= Path(__file__).parent / "training" / "sft.yaml"
    )


    rm_train_func = Component.from_yaml(
        ws,
        yaml_file= Path(__file__).parent / "training" / "rm.yaml"
    )


    ppo_func = Component.from_yaml(
        ws,
        yaml_file= Path(__file__).parent / "training" / "ppo.yaml"
    )


    @dsl.pipeline(
        name=f"{timestamp} {TASK} {model}",
        default_compute_target=default_compute_target,
        default_datastore='workspaceblobstore',
    )
    def train_pipeline():
        
        # '''
        if sft_model_weights_input is None:
            trainer = train_func(
                data_path=all_datasets_path,
                # data_split=data_split,
                data_split="1,0,0",
                model_dir=model_dir,
                model_name_or_path=model_name_or_path,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                max_seq_len=max_seq_len,
                learning_rate=sft_lr,
                weight_decay=sft_wd,
                num_train_epochs=sft_e,
                # gradient_accumulation_steps=1,
                gradient_accumulation_steps=gradient_accumulation_steps,
                lr_scheduler_type="cosine",
                num_warmup_steps=0,
                seed=42,
                zero_stage=zero_stage,
                lora_dim=sft_lora,
                lora_module_name=lora_module_name,
            )
            trainer.runsettings.resource_layout.configure(instance_count=instance_count, process_count_per_node=process_count_per_node)

            trainer.outputs.output_dir.configure(
                mode="mount",
                path_on_datastore=output_path + "/sft",
                datastore=ds
            )

            sft_model_weights = trainer.outputs.output_dir
        else:
            sft_model_weights = sft_model_weights_input

        # '''

        # '''
        if rm_model_weights_input is None and reward_model_name_or_path is None:
            rm_per_device_train_batch_size = per_device_train_batch_size    

            rm_trainer = rm_train_func(
                data_path=all_datasets_path,
                data_split="0,1,0",
                model_dir=model_dir,
                model_name_or_path=model_name_or_path,
                per_device_train_batch_size=rm_per_device_train_batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                max_seq_len=max_seq_len,
                num_padding_at_beginning=num_padding_at_beginning,
                learning_rate=rm_lr,
                weight_decay=rm_wd,
                num_train_epochs=rm_e,
                gradient_accumulation_steps=rm_gradient_accumulation_steps,
                lr_scheduler_type="cosine",
                num_warmup_steps=0,
                seed=42,
                zero_stage=zero_stage,
                lora_dim=rm_lora,
                lora_module_name=lora_module_name,
            )
            rm_trainer.runsettings.resource_layout.configure(instance_count=instance_count, process_count_per_node=process_count_per_node)

            rm_trainer.outputs.output_dir.configure(
                mode="mount",
                path_on_datastore=output_path + "/rm",
                datastore=ds
            )

            rm_model_weights = rm_trainer.outputs.output_dir
        else:
            rm_model_weights = rm_model_weights_input

            

        
        # '''   ppo without sft

        if ppo_baseline_input is None:
            ppo = ppo_func(
                data_path=all_datasets_path,
                data_split="0,0,1",

                actor_model_dir=None,
                actor_model_name_or_path=model_name_or_path,
                critic_model_dir=rm_model_weights,
                critic_model_name_or_path=None,

                num_padding_at_beginning=num_padding_at_beginning,
                per_device_generation_batch_size=per_device_train_batch_size,
                per_device_training_batch_size=per_device_mini_train_batch_size,
                generation_batches= 1,
                ppo_epochs=1,
                max_answer_seq_len=max_answer_seq_len,
                max_prompt_seq_len=max_prompt_seq_len,

                actor_lora_learning_rate=actor_lora_learning_rate,
                critic_lora_learning_rate=critic_lora_learning_rate,

                actor_weight_decay=0,
                critic_weight_decay=0,

                num_train_epochs=1,
                lr_scheduler_type="cosine",
                gradient_accumulation_steps=ppo_gradient_accumulation_steps,
                num_warmup_steps=100,
                seed=42,
                actor_zero_stage=ppo_zero_stage,
                critic_zero_stage=ppo_zero_stage,
                actor_lora_dim=128,
                actor_lora_module_name=lora_module_name,
                critic_lora_dim=128,
                critic_lora_module_name=lora_module_name,

            )

            ppo.runsettings.resource_layout.configure(instance_count=ppo_instance_count, process_count_per_node=process_count_per_node)

            ppo.outputs.output_dir.configure(
                mode="mount",
                path_on_datastore=output_path + "/ppo",
                datastore=ds
            )
            

            ppo_baseline = ppo.outputs.output_dir
        else:
            ppo_baseline = ppo_baseline_input

        # '''



        # ''' normal

        if ppo_baseline_input is None:
            ppo = ppo_func(
                data_path=all_datasets_path,
                data_split="0,0,1",

                actor_model_dir=sft_model_weights,
                actor_model_name_or_path=None,
                critic_model_dir=rm_model_weights,
                critic_model_name_or_path=None,
                num_padding_at_beginning=num_padding_at_beginning,
                per_device_generation_batch_size=per_device_train_batch_size,
                per_device_training_batch_size=per_device_mini_train_batch_size,
                generation_batches= 1,
                ppo_epochs=1,
                max_answer_seq_len=max_answer_seq_len,
                max_prompt_seq_len=max_prompt_seq_len,

                actor_lora_learning_rate=actor_lora_learning_rate,
                critic_lora_learning_rate=critic_lora_learning_rate,

                actor_weight_decay=0,
                critic_weight_decay=0,

                num_train_epochs=1,
                lr_scheduler_type="cosine",
                gradient_accumulation_steps=ppo_gradient_accumulation_steps,
                num_warmup_steps=100,
                seed=42,
                actor_zero_stage=ppo_zero_stage,
                critic_zero_stage=ppo_zero_stage,
                actor_lora_dim=128,
                actor_lora_module_name=lora_module_name,
                critic_lora_dim=128,
                critic_lora_module_name=lora_module_name,

            )

            ppo.runsettings.resource_layout.configure(instance_count=ppo_instance_count, process_count_per_node=process_count_per_node)

            ppo.outputs.output_dir.configure(
                mode="mount",
                path_on_datastore=output_path + "/ppo",
                datastore=ds
            )
            

            ppo_baseline = ppo.outputs.output_dir
        else:
            ppo_baseline = ppo_baseline_input

        # '''



        '''    for pythia rward

        if ppo_baseline_input is None:
            ppo = ppo_func(
                data_path=all_datasets_path,
                data_split="0,0,1",

                actor_model_dir=sft_model_weights,
                actor_model_name_or_path=None,
                critic_model_dir=sft_model_weights,
                critic_model_name_or_path=None,

                reward_model_dir = None,
                reward_model_name_or_path = reward_model_name_or_path,
                
                num_padding_at_beginning=num_padding_at_beginning,
                per_device_generation_batch_size=per_device_train_batch_size,
                per_device_training_batch_size=per_device_mini_train_batch_size,
                generation_batches= 1,
                ppo_epochs=1,
                max_answer_seq_len=max_answer_seq_len,
                max_prompt_seq_len=max_prompt_seq_len,

                actor_lora_learning_rate=actor_lora_learning_rate,
                critic_lora_learning_rate=critic_lora_learning_rate,

                actor_weight_decay=0,
                critic_weight_decay=0,

                num_train_epochs=1,
                lr_scheduler_type="cosine",
                gradient_accumulation_steps=ppo_gradient_accumulation_steps,
                num_warmup_steps=100,
                seed=42,
                actor_zero_stage=ppo_zero_stage,
                critic_zero_stage=ppo_zero_stage,
                actor_lora_dim=128,
                actor_lora_module_name=lora_module_name,
                critic_lora_dim=128,
                critic_lora_module_name=lora_module_name,

            )

            ppo.runsettings.resource_layout.configure(instance_count=ppo_instance_count, process_count_per_node=process_count_per_node)

            ppo.outputs.output_dir.configure(
                mode="mount",
                path_on_datastore=output_path + "/ppo",
                datastore=ds
            )
            

            ppo_baseline = ppo.outputs.output_dir
        else:
            ppo_baseline = ppo_baseline_input

        # '''

    pipeline = train_pipeline()
    _ = pipeline.submit(
        experiment_name="misantac_oss_rlhf_sep_rm",
        # continue_on_step_failure=True,
    )


if __name__ == "__main__":
    main()
