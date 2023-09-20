# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
)
from huggingface_hub import snapshot_download
from transformers.deepspeed import HfDeepSpeedConfig

from .reward_model import RewardModel, GPTNeoXRewardModel
from ..utils import load_state_dict_into_model


def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    config_only=False,
                    disable_dropout=False):
    
    if False:
        model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        if disable_dropout:
            model_config.dropout = 0.0
        # Note: dschf is defined in function scope to avoid global effects
        # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            dschf = HfDeepSpeedConfig(ds_config)
        else:
            dschf = None
        if config_only:
            # the weight loading is handled by create critic model
            model = model_class.from_config(model_config)
        else:
            model = model_class.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=model_config,
                trust_remote_code=True)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            dschf = HfDeepSpeedConfig(ds_config)
        else:
            dschf = None

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    # model.resize_token_embeddings(int(
        # 8 *
        # math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    # if config_only:

    return model


def create_critic_model(model_name_or_path,
                        tokenizer,
                        ds_config,
                        num_padding_at_beginning=0,
                        rlhf_training=False,
                        disable_dropout=False,
                        zero_stage=0):
    # OPT model family always put a padding token at the beginning of the sequence,
    # we did not see this in other models but not sure if it is a general rule

    import time

    start = time.time()
    seq_class = False
    try:
        base_critic_model = create_hf_model(AutoModel, model_name_or_path, tokenizer,
                                    ds_config, rlhf_training, disable_dropout)
    except:
        base_critic_model = create_hf_model(AutoModelForSequenceClassification, model_name_or_path, tokenizer,
                                    ds_config, rlhf_training, disable_dropout)
        seq_class = True
        
    from .reward_model import PhiTransformer
    base_critic_model = PhiTransformer(base_critic_model.layers)
    config = AutoConfig.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)

    end = time.time()
    if torch.distributed.get_rank() == 0:
        print(f"> Creating model from_config took {end - start} seconds")
    
    critic_model = base_critic_model
    print("Loaded critic model as type", type(base_critic_model))

    v_head_weights = None
    if isinstance(base_critic_model, GPTNeoXRewardModel) or seq_class:
        critic_model = base_critic_model.gpt_neox
        v_head_weights = base_critic_model.out_proj.weight

    critic_model = RewardModel(
        critic_model,
        tokenizer,
        num_padding_at_beginning=num_padding_at_beginning,
        v_head_weights=v_head_weights,
        config=config)

    if rlhf_training:
        if not os.path.isdir(model_name_or_path):
            # we don't know if the model is in one file or multiple files, so we must download from hf hub
            start = time.time()
            try:
                model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
            except:
                model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, trust_remote_code=True)
            model_ckpt_state_dict = model.state_dict()
            end = time.time()
            if torch.distributed.get_rank() == 0:
                print(f"> model download took {end - start} seconds")

            if isinstance(base_critic_model, GPTNeoXRewardModel):
                model_ckpt_state_dict = fix_state_dict_keys(model_ckpt_state_dict)

                print("critic model state dict keys")
                print(critic_model.state_dict().keys())

        else:
            # load critic model from checkpoint
            model_ckpt_path = os.path.join(model_name_or_path, 'pytorch_model.bin')
            assert os.path.exists(
                model_ckpt_path
            ), f"Cannot find model checkpoint at {model_ckpt_path}"

            start = time.time()
            model_ckpt_state_dict = torch.load(model_ckpt_path, map_location='cpu')
            end = time.time()
            if torch.distributed.get_rank() == 0:
                print(f"> torch.load took {end - start} seconds")

        

        # load critic model from checkpoint with zero-stage 3 compatibility
        # this functionality may be moved to DS checkpoint load API in future
        start = time.time()
        missing_keys, unexpected_keys, error_msgs = load_state_dict_into_model(critic_model,
                                   model_ckpt_state_dict,
                                   "",
                                   zero_stage=zero_stage)
        end = time.time()

        print("missing keys")
        print(missing_keys)
        print("unexpected keys")
        print(unexpected_keys)
        print("error msgs")
        print(error_msgs)
        if torch.distributed.get_rank() == 0:
            print(f"> Loading model state dict took {end - start} seconds")

    return critic_model



def fix_state_dict_keys(model_ckpt_state_dict):
        print("model_ckpt_state_dict keys", model_ckpt_state_dict.keys())

        new_state_dict = {}

        for key in model_ckpt_state_dict.keys():
            if "model" in key:
                new_state_dict[key.replace("model", "rwtranrsformer")] = model_ckpt_state_dict[key]
            elif "gpt_neox" in key:
                new_state_dict[key.replace("gpt_neox", "rwtranrsformer")] = model_ckpt_state_dict[key]
            elif "score" in key:
                new_state_dict[key.replace("score", "v_head")] = model_ckpt_state_dict[key]
            elif "out_proj" in key:
                new_state_dict[key.replace("out_proj", "v_head")] = model_ckpt_state_dict[key]
            else:
                new_state_dict[key] = model_ckpt_state_dict[key]

        print("new state dict keys", new_state_dict.keys())
        model_ckpt_state_dict = new_state_dict
        
        return model_ckpt_state_dict