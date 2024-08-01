#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
from typing import Optional, Dict
import torch
import transformers
from transformers import Trainer
import os
from utils.data_utils import prepare_encoded_data, load_encoded_data

prompter = """[INST] For a given question, your task is to parse the question into a correct logical form (s-expression) which could be executed over a KG to get the answer. To write the s-expression, you need to decompose the question into subtasks and solve them step-by-step. To get necessary schemas (i.e., relations or classes) for s-expression writing, you can use the following functions to interact with the KG.\nget_relations(expression): return all relations of those entities represented by the expression.\nget_classes(expression): return all class types of those entities represented by the expression.\nget_relevant_relations(thought): retrieve relevant relations from the KG according to your thought.\nget_relevant_classes(thought): retrieve relevant classes from the KG according to your thought.\nget_descriptions(candidate): get description of candidate schemas.\nThe question is {input} [/INST] The given question can be decomposed into the following subtasks:"""

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="llamas", metadata={"help": "Path to the pre-trained model."})

@dataclass
class DataArguments:
    data_dir: str = field(default="", metadata={"help": "Path to the training data."})
    cutoff_len: int = field(default=2048, metadata={"help": "Maximum sequence length."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    wandb_project: str = field(default="", metadata={"help": "Name of the wandb project."})



def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg



def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if len(training_args.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project

    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path),
    print('config...', config)
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config = config
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    tokenizer.pad_token_id = (
        0
    )

    def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
    ):
        """Resize tokenizer and embedding.

        Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
        """
        num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

        if num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

    # if processed_data doesn't exist, process the data otherwise, load the processed data
    encoded_data_path = os.path.join(data_args.data_dir, 'encoded_data')
    if not os.path.exists(encoded_data_path):
        print('convert data to encoded data...')
        train_data = prepare_encoded_data(data_args.data_dir, prompter, encoded_data_path, data_args.cutoff_len, tokenizer)
    else:
        train_data = load_encoded_data(os.path.join(encoded_data_path, 'train_data'))

    train_data = train_data.shuffle()


    example = train_data[0:4]
    print(tokenizer.batch_decode(example['input_ids']))

    for i in range(4):
        print(tokenizer.decode(torch.tensor(example['input_ids'][i])))
    print(f'the length of training dataset is {len(train_data)}')


    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )  # Be more transparent about the % of trainable params.
    
    print_trainable_parameters(model)


    trainer = Trainer(model=model,train_dataset=train_data, eval_dataset=None, tokenizer=tokenizer, args=training_args, data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True))

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()