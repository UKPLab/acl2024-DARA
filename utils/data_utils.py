from datasets import load_from_disk, load_dataset, Dataset
from typing import Union
from transformers import AutoTokenizer
import os
import pandas as pd
import re


prompter = """[INST] For a given question, your task is to parse the question into a correct logical form (s-expression) which could be executed over a KG to get the answer. To write the s-expression, you need to decompose the question into subtasks and solve them step-by-step. To get necessary schemas (i.e., relations or classes) for s-expression writing, you can use the following functions to interact with the KG.\nget_relations(expression): return all relations of those entities represented by the expression.\nget_classes(expression): return all class types of those entities represented by the expression.\nget_relevant_relations(thought): retrieve relevant relations from the KG according to your thought.\nget_relevant_classes(thought): retrieve relevant classes from the KG according to your thought.\nget_descriptions(candidate): get description of candidate schemas.\nThe question is {input} [/INST] The given question can be decomposed into the following subtasks:"""


def load_encoded_data(encoded_data_path):
    return load_from_disk(encoded_data_path)


def tokenize(prompt, tokenizer, cutoff_len, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )

    if (
        result["input_ids"][0][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        for i in range(len(result["input_ids"])):
            result["input_ids"][i].append(tokenizer.eos_token_id)
            result["attention_mask"][i].append(1)
        
    result["labels"] = result["input_ids"].copy()
    return result


def generate_tokenize_prompt(data_point, prompter, cutoff_len, tokenizer):

    full_prompt = prompter.format(
        input = data_point['input'])
    
    if data_point["output"]:
        full_prompt = full_prompt + "\n".join(data_point["output"])
 
    tokenized_full_prompt = tokenize([full_prompt], tokenizer=tokenizer, cutoff_len = cutoff_len)
    return tokenized_full_prompt


def prepare_encoded_data(data_dir: Union[str, list], prompter: str, encoded_data_path:str, cutoff_len:int, tokenizer: AutoTokenizer):
    data_path = os.path.join(data_dir, 'reasoning_chain.json')
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path, split="train")
    
    encoded_train_path = os.path.join(encoded_data_path, "train_data")

    encoded_data = data.shuffle().map(lambda x: generate_tokenize_prompt(x, prompter=prompter, cutoff_len=cutoff_len, tokenizer=tokenizer))
    encoded_data = pd.DataFrame(encoded_data).explode(['input_ids', 'attention_mask', 'labels'])
    
    subset = encoded_data[['input_ids', 'labels']][0:2000]
    subset['label_text'] = tokenizer.batch_decode(subset['input_ids'])
    encoded_data = Dataset.from_pandas(encoded_data)

    encoded_data.save_to_disk(encoded_train_path)
    subset.to_csv(os.path.join(encoded_data_path, 'train_data_example.csv'))
    
    return encoded_data


def post_process_predictions_react(s_expression, s_exp_dict, reverse_properties_dict, s_expression_extractor=re.compile('s-exp-[0-9.]+'), webqa=False, http_remover=re.compile('\^\^http(:\/\/www\.w3\.org\/[0-9a-zA-Z\/#]+)?')):
    '''replace s-exp-xxx with the corresponding s-expression'''
    try:
        s_exp_ids = s_expression_extractor.findall(s_expression)
    except:
        print(s_expression)
        return
    if webqa:
        if s_exp_ids:
            s_expression = s_exp_ids[0]
        
    if s_exp_ids:
        try:
            for s_exp_id in s_exp_ids:
                s_expression = s_expression.replace(s_exp_id, s_exp_dict[s_exp_id])
        except:
            pass
    return s_expression
