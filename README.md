# DARA: Decomposition-Alignment-Reasoning Autonomous Language Agent for Question Answering over Knowledge Graphs
[![License](https://img.shields.io/github/license/UKPLab/ukp-project-template)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)


<p align="center">
    🤗 <a href="https://huggingface.co/UKPLab/dara-mistral-7b" target="_blank">Models</a> |  🤗 <a href="https://huggingface.co/datasets/UKPLab/dara" target="_blank">Dataset</a> | 📃 <a href="https://arxiv.org/abs/2406.07080" target="_blank">Paper</a>
</p>

This repository implements the DARA, a LLM-based agent for KGQA, as described in [DARA: Decomposition-Alignment-Reasoning Autonomous Language Agent for Question Answering over Knowledge Graphs](https://aclanthology.org/2024.findings-acl.203/) 

> **Abstract** : Answering Questions over Knowledge Graphs (KGQA) is key to well-functioning autonomous language agents in various real-life applications. To improve the neural-symbolic reasoning capabilities of language agents powered by Large Language Models (LLMs) in KGQA, we propose the DecompositionAlignment-Reasoning Agent (DARA) framework. DARA effectively parses questions into formal queries through a dual mechanism: high-level iterative task decomposition and low-level task grounding. Importantly, DARA can be efficiently trained with a small number of high-quality reasoning trajectories. Our experimental results demonstrate that DARA fine-tuned on LLMs (e.g. Llama-2-7B, Mistral) outperforms both in-context learning-based agents with GPT-4 and alternative fine-tuned agents, across different benchmarks in zero-shot evaluation, making such models more accessible for real-life applications. We also show that DARA attains performance comparable to state-of-the-art enumerating-and-ranking-based methods for KGQA.

Contact person: [Haishuo Fang](mailto:haishuo.fang@tu-darmstadt.de) 

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/
)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.


## 🚀 Setup
- Environment
```sh
> python -m venv .dara
> source ./.dara/bin/activate
> pip install -r requirements.txt
```
- Freebase Setup
To install Freebase, please refer to [here](https://github.com/shijx12/KQAPro_Baselines/tree/master/SPARQL)

Once the service is up, replace the url in the `line 6 of ./kg_querier/sparql_executor.py`
```python
sparql = SPARQLWrapper("url/to/service")
```
## Models
|  Models            | Training Data | Model Card|
|--------------------|---------------|-----------|
|  DARA-Llama-2-7B   | [UKPLab/dara](https://huggingface.co/datasets/UKPLab/dara) |[UKPLab/dara-llama-2-7b](https://huggingface.co/UKPLab/dara-llama-2-7b)|
|DARA-Llama-2-13B|[UKPLab/dara](https://huggingface.co/datasets/UKPLab/dara) | [UKPLab/dara-llama-2-13b](https://huggingface.co/UKPLab/dara-llama-2-13b)|
|  DARA-Mistral-7B        |[UKPLab/dara](https://huggingface.co/datasets/UKPLab/dara)| [UKPLab/dara-mistral-7b](https://huggingface.co/UKPLab/dara-mistral-7b)|
|Agentbench-7B| [UKPLab/dara-agentbench](https://huggingface.co/datasets/UKPLab/DARA-Agentbench)|[UKPLab/agentbench-7b](https://huggingface.co/UKPLab/agentbench-7b)|


## Fine-tuning
```sh
torchrun --nproc_per_node=2 --master_port=8889 finetune.py \
    --model_name_or_path /path/to/model \
    --data_dir ./data/finetune_data/dara.json \
    --output_dir ./models/$1 \
    --wandb_project kgqa-dara \
    --run_name $2 \
    --report_to wandb \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --logging_steps 20 \
    --cutoff_len 2500 \
    --learning_rate 2e-6 \
    --lr_scheduler_type cosine \
    --save_total_limit 4 \
    --weight_decay 0.00 \
    --warmup_ratio 0.1 \
    --tf32 True \
    --bf16 True \
    --deepspeed "./configs/default_offload_opt_param.json" \
    --gradient_checkpointing True \
```
As we use deepspeed to fine-tune the model, `python zero_to_fp32.py . pytorch_model.bin` is needed to convert the model format.

## Generation

```sh
if [ $1 == "dara" ]; then
    python generate.py \
    --pred_file_path ./data/test_data/$2.json \
    --base_model $3 \
    --num_q $4 \
    --start_ix $5 \
    --output_dir ./outputs/dara/$6 \
    --batch_size 1 \

elif [ $1 == "agentbench" ]; then
    python -m baseline.agentbench \
    --pred_file_path ./data/test_data/$2.json \
    --base_model $3 \
    --num_q $4 \
    --start_ix $5 \
    --ouptut_dir ./outputs/agentbench/$6 \
    --batch_size 1 \
    --use_gpt \

fi
```

## Evaluation
The evalaution datasets are under `./test_data/`
```sh
python evaluate.py \
--gold_data_path ./data/test_data/grailqa.json \
--predict_data_dir ./outputs/dara \
--metric_output_path ./eval/grailqa/ \
```

## Cite
```
@inproceedings{fang-etal-2024-dara,
    title = "$\texttt{DARA}$: Decomposition-Alignment-Reasoning Autonomous Language Agent for Question Answering over Knowledge Graphs",
    author = "Fang, Haishuo  and
      Zhu, Xiaodan  and
      Gurevych, Iryna",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.203",
    pages = "3406--3432"}
}
```

## Disclaimer
> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
