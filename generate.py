import os
import sys
import re
import fire
import torch
import transformers
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from kg_querier.kg_querier import KGQuerier

import json
from tqdm import trange
from utils.logitis_processor import _logitsProcessor
import openai

prompter = """[INST] For a given question, your task is to parse the question into a correct logical form (s-expression) which could be executed over a KG to get the answer. To write the s-expression, you need to decompose the question into subtasks and solve them step-by-step. To get necessary schemas (i.e., relations or classes) for s-expression writing, you can use the following functions to interact with the KG.\nget_relations(expression): return all relations of those entities represented by the expression.\nget_classes(expression): return all class types of those entities represented by the expression.\nget_relevant_relations(thought): retrieve relevant relations from the KG according to your thought.\nget_relevant_classes(thought): retrieve relevant classes from the KG according to your thought.\nget_descriptions(candidate): get description of candidate schemas.\nThe question is {input} [/INST] The given question can be decomposed into the following subtasks:"""

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

class KGQAAgent(object):
    def __init__(self,
        load_8bit: bool = False,
        base_model: str = "",
        use_gpt: bool = False
    ):

        base_model = base_model or os.environ.get("BASE_MODEL", "")
        assert (
            base_model
        ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
        self.use_gpt = use_gpt
        self.pad_token = '<unk>'

        if not self.use_gpt:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir = "cache"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
            self.model.config.bos_token_id = 1
            self.model.config.eos_token_id = 2
            self.tokenizer.padding_side = "left"

            if not load_8bit:
                self.model.half()  # seems to fix bugs for some users.
            
            self.model.eval()
            if torch.__version__ >= "2" and sys.platform != "win32":
                self.model = torch.compile(self.model)
            
        else:
            self.model = None

    def reasoning(self, generation_config, max_new_tokens, input_q, kg_querier, action_input_extactor, s_exp_extractor, stop_words):

        with torch.no_grad():
            if not self.use_gpt:
                logit_processor = _logitsProcessor(batch_size=len(input_q), stop_words=stop_words, tokenizer=self.tokenizer, device=device)
                LogitsProcessorList = transformers.LogitsProcessorList([logit_processor])
            # print(stop_words_ids.sentinel_token_ids)
                generate_params = {
                    "generation_config": generation_config,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                    "max_new_tokens": max_new_tokens,
                    "logits_processor": LogitsProcessorList,
                    "pad_token_id": self.tokenizer.pad_token_id
                }
                prompts = []
                prompts = [prompter.format(input=prompt) for prompt in input_q]
                seen_sents = [prompt.split('\n') for prompt in prompts]

            else:
                prompts = ["" for _ in input_q]
                with open("./icl_dara_gpt4.json", "r") as f:
                    message = json.load(f)
                seen_sents = [[] for _ in range(len(input_q))]

            s_expression = [{} for _ in range(len(input_q))]
            exec_result = [None for _ in range(len(input_q))]
            unfinished_questions = [True]*len(input_q)
            overall_plan = ["" for _ in range(len(input_q))]
            step_num = 0
            remaining_q = []
            # Note: for the generate prompt, it is possible to generate something new after self.PAD_TOKENself.PAD_TOKEN... since generation has been called multiple times
            while True:
                if self.use_gpt:
                    for i, q in enumerate(input_q):
                        if step_num == 0:
                                message.append({"role": "user", "content": f"Great! The new question is '{q}'"})
                        else:
                            if prompts[i] == "": continue
                            message[-1]['content'] = prompts[i]
                        data = {'model': 'gpt-4', 'messages': message, 'temperature': 0.75, 'max_tokens': 300, 'stop': ['### Obs']}
                        # res = requests.post(url, headers=headers, json=data).json()
                        try:
                            res = openai.ChatCompletion.create(engine="gpt-4", **data)
                        except openai.error.InvalidRequestError:
                            if q.split('\n')[0] not in remaining_q:
                                remaining_q.append(q.split('\n')[0])
                            continue
                    # seperate them due to 'content_filter' and dont wanna skip due to timeout
                        try:
                        # print(res)
                            if message[-1]['role'] == 'user':
                                message.append({"role":"assistant", "content":res['choices'][0]['message']['content']})
                            prompts[i] += res['choices'][0]['message']['content'] + '\n'
                        except:
                            if q.split('\n')[0] not in remaining_q:
                                remaining_q.append(q.split('\n')[0])
                            continue
                else:
                    inputs = self.tokenizer(prompts, padding="longest", return_tensors="pt")
                    input_ids = inputs["input_ids"].to(device)
                    attention_masks = inputs["attention_mask"].to(device)

                    generate_params["input_ids"] = input_ids
                    generate_params["attention_mask"] = attention_masks
                    generation_output = self.model.generate(**generate_params)
                    s = generation_output.sequences
                    prompts = self.tokenizer.batch_decode(s.detach().cpu())

                # ipdb.set_trace()
                for ix, prompt in enumerate(prompts):
                    # prompt = prompt.outputs[0].text
                    if not unfinished_questions[ix]: continue
                    # used to interact with KG
                    s_expression[ix], prompts[ix], seen_sents[ix], exec_result[ix], overall_plan[ix] = self.step_reasoning(kg_querier, action_input_extactor, s_exp_extractor, s_expression[ix], seen_sents[ix], prompt, input_q[ix], overall_plan[ix])

                    #used to identify whether the loop is finished
                    if prompts[ix].endswith('</s>') or prompts[ix].endswith(self.pad_token) or 'Final s-exp' in s_expression[ix]:
                        # ipdb.set_trace()
                        unfinished_questions[ix] = False
                        # plan = prompts[ix].split('\n#')[0]

                if not any(unfinished_questions):
                    break
                step_num += 1
                if step_num > 6:
                    break
            return s_expression, prompts, exec_result, overall_plan

    def step_reasoning(self, kg_querier, action_input_extactor, s_exp_extractor, s_expression, seen_sents, prompt, question, overall_plan):
        exec_result = 'success'
        sentences = prompt.split('\n')
        # remove pad_token introduced by batch generation
        # stop_type = sentences[-1].replace('</s>', '').replace(self.pad_token, '')
        curr_sents_list = []
        # new_s_exp at this generation step
        new_s_exp = ""

        for sent in sentences:
            # in case it starts with whitespace
            sent = sent.strip()
            if sent in seen_sents or '### Obs</s>' in sent: continue
            # get plan
            if (sent.startswith('### S-exp-') or sent.startswith('## S-exp-')) and ('Action' not in sent):
                try:
                    key = sent.split(':', 1)[0].strip().lower()
                    key = key.replace('#', '').strip()
                    new_s_exp = sent.split(':', 1)[1].strip().strip('.').replace('</s>','').replace(self.pad_token,'')
                    for k in s_expression:
                        if k in new_s_exp:
                            new_s_exp = new_s_exp.replace(k, s_expression[k])
                    # verify the s-exp
                    # status = s_exp_verification(s_exp)
                    #
                    s_expression[key] = new_s_exp
                except:
                    print(f'when extracting s-exp something wrong with {sent}')
            curr_sents_list.append(sent)
        seen_sents.extend(curr_sents_list)
        # ipdb.set_trace()
        if not curr_sents_list:
            return s_expression, prompt, seen_sents, exec_result, overall_plan
        
        curr_sents = "\n".join(curr_sents_list)
        # ipdb.set_trace()
        if 'Final s-exp' in curr_sents:
            try:
                s_exp = prompt.rsplit("Final s-exp:", 1)[1].replace('\n', '')
                s_expression["Final s-exp"] = s_exp.replace('</s>','').replace(self.pad_token,'')
            except:
                print('no right side of the final s-exp')
                exec_result = 'no right side of the final s-exp, check the prompt'
            return s_expression, prompt, seen_sents, exec_result, overall_plan
        # For verification
        # print(f'stop_type is {stop_type}')
        # print('prompt is ', prompt)

        try:
        # get action: 1.1: get_relations(m.05pwqjc)\nObs
            # ipdb.set_trace()
            action = curr_sents.rsplit('Action', 1)[1]
            # remove \nObs
            action = action.replace('\nObs', '')
            # 1.1
            order = action.split(':', 1)[0].strip()
            action_content = action.split(':', 1)[1].strip()
            # if 'KG_executor' in action_content:
            action_content = action_input_extactor.search(action_content)
            func_name = action_content.group(1)
            input = action_content.group(2)
            # replace CONS with AND
            if '(CONS' in input:
                componets = input.split(' ')
                entity_or_str = componets[3].strip(')')
                if not (entity_or_str.startswith('m.') or entity_or_str.startswith('g.')):
                    entity_or_str = f'{entity_or_str}'
                input = f"(AND {componets[1]} (JOIN {componets[2]} {entity_or_str}))"
        except:
            print('Finished! No action found..')
            exec_result = f'Finished! No action found, check the prompt'
            return s_expression, prompt, seen_sents, exec_result, overall_plan
        # action_content = action_content.replace('KG_executor(', '').replace(')', '')
        # Retrive relevant classes or relations from FB
        try:
            if func_name == 'get_relevant_classes':
                result = kg_querier.get_relevant_classes_from_FB(query=input)
            
            elif func_name == 'get_relevant_relations':
                result = kg_querier.get_relevant_relations_from_FB(query=input, question=question.split('The linked')[0])
            
            elif func_name == 'get_descriptions':
                result = kg_querier.get_descriptions_from_FB(relation=input)
            
            elif func_name == 'get_classes' or func_name == 'get_relations':
                if 's-exp' in input.lower():
                    s_exp = s_exp_extractor.findall(input)
                    assert len(s_exp) == 1
                    s_exp = s_exp[0].lower()
                    input = input.replace(s_exp, s_expression[s_exp])
                    var_name = s_exp
                    query = input
                else:
                    var_name = input
                    query = input
                if func_name == 'get_classes':
                    result = kg_querier.get_classes_from_FB(var_name=var_name, question=question.split('The linked')[0], query=query)
                
                elif func_name == 'get_relations':
                    result = kg_querier.get_two_stages_relations_from_FB(var_name=var_name, question=question.split('The linked')[0], query=query)
            
            elif func_name == 'verify_query':  
                s_exp = s_exp_extractor.findall(input.lower())
                assert len(s_exp) == 1
                s_exp = s_exp[0] 
                result = kg_querier.verify_query(s_expression[s_exp])

        except Exception as e:
            # ipdb.set_trace()
            print('Interacting with KG failed: ', e)
            # print(f'Pass this question {question}. Query:{query}. Var_name:{var_name}. No result found..')
            exec_result = f'Interacting with KG failed. The func is {func_name}. Query is {input}. No result found.'
            return s_expression, prompt, seen_sents, exec_result, overall_plan
        # for those actions that don't exist in the above four, we just skip them
        if result:
            prompt = prompt.replace("</s>","").replace(self.pad_token, "").strip()
            if prompt.endswith('### Obs'):
                prompt += f' {order}: ' + result
            else:
                prompt += f'### Obs {order}: ' + result

        return s_expression, prompt, seen_sents, exec_result, overall_plan

    def action(self,
        questions=None,
        temperature=0.0,
        top_p=1,
        top_k=50,
        do_sample=False,
        max_new_tokens=500,
        batch_size=10,
        prediction_file='./outputs/action_results.txt',
        stop_words=None,
        **kwargs,
    ):
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            **kwargs,
        )
        
        # Without streaming
        action_input_extactor = re.compile('(get_relations|get_relevant_classes|get_relevant_relations|get_classes|get_descriptions)\((.*)\)')
        s_exp_extractor = re.compile('[Ss]-exp-[0-9.]+')
        kg_querier = KGQuerier()

            
        transformers.set_seed(49)
        fw = open(prediction_file, 'a', encoding='utf-8')
        # ipdb.set_trace()
        for ix in trange(0, len(questions), batch_size):
            batch_question = questions[ix:ix+batch_size]
            s_expression, prompts, exec_result, overall_plan = self.reasoning(generation_config, max_new_tokens, batch_question, kg_querier, action_input_extactor, s_exp_extractor, stop_words)
            # ipdb.set_trace()
            for qid in range(len(batch_question)):
                fw.write(json.dumps({'question': batch_question[qid], 'exec_result': exec_result[qid], 'overall_plan': overall_plan[qid], 's_expression':s_expression[qid], 'prompt': prompts[qid]})+'\n')
            fw.flush()
        fw.close()
        return
    
def main(start_ix, num_q, pred_file_path, batch_size, base_model, load_8bit: bool = False, output_dir='./outputs/action_results.txt', use_gpt=False):
    '''
    start_ix: int, the starting index of the questions to be processed
    num_q: int, the number of questions to be processed
    pred_file_path: str, the path to the test file
    batch_size: int, the batch size for processing the questions
    base_model: str, the fine-tuned model to be used
    load_8bit: bool, whether to load the model in 8-bit
    output_dir: str, the directory to save the output
    use_gpt: bool, whether to use gpt for reasoning
    '''
    
    agent = KGQAAgent(load_8bit=load_8bit, base_model=base_model, use_gpt=use_gpt)

    with open(pred_file_path, 'r') as f:
        data = json.load(f)

    if 'input' not in data[0]:
        questions = [d['question'] for d in data[start_ix:start_ix+num_q]]
    else:
        questions = [d['input'] for d in data[start_ix:start_ix+num_q]]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('output_dir', output_dir)
    print('start_ix', start_ix)
    output_file = os.path.join(output_dir, f'start_{start_ix}.txt')
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_data = f.readlines()
        old_questions = [json.loads(line)['question'] for line in existing_data]       
        questions = [q for q in questions if q not in old_questions]
    
    debug = False

    if debug:
        # what unit of fuel economy has economy in litres per kilometre less than 0.01
        #volts is in which measurement system? The linked entity is volts (m.07_7_).
        test_data = ["which is the unit of time in the international system of units used to measure less than 1000.0 seconds? The linked entity is International System of Units (m.0c13h)."]
        agent.action(questions = test_data, stop_words=['### Obs'], batch_size=batch_size, prediction_file=output_file, use_gpt=use_gpt)
    else:
        agent.action(questions = questions, stop_words=['### Obs'], batch_size=batch_size, prediction_file=output_file, use_gpt=use_gpt)

if __name__ == "__main__":
    openai.api_key = "e086b9858984434d8b70f537d8574002"
    openai.api_base = "https://azure-openai-ukp-004.openai.azure.com/" # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
    openai.api_type = 'azure'
    openai.api_version = '2023-05-15' # this may change in the future
    agent = fire.Fire(main)