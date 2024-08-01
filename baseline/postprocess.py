from kg_querier.sparql_executor import execute_query
from kg_querier.logic_form_util import lisp_to_sparql
import json
import os
from tqdm import tqdm
import sys

if __name__ == '__main__':
    dataset = sys.argv[1]
    model = sys.argv[2]
    if dataset == 'graphqa':
        predict_dir = f'./outputs/icl_dara/{model}/graphqa/'
        post_processed_dir = f'./outputs/icl_dara/{model}/graphqa/processed'
        test_data = './data/evaluate_data/graphqa/test_200.json'
    elif dataset == 'grailqa':
        predict_dir = f'./outputs/icl_dara/{model}/grailqa/'
        post_processed_dir = f'./outputs/icl_dara/{model}/grailqa/processed'
        test_data = './data/evaluate_data/grailqa/test_200.json'
    elif dataset =='webqa':
        predict_dir = f'./outputs/icl_dara/{model}/webqa/'
        post_processed_dir = f'./outputs/icl_dara/{model}/webqa/processed'
        test_data = './data/evaluate_data/webqa/test.json'
    
    files = os.listdir(predict_dir)
    predictions = []
    for file in files:
        with open(os.path.join(predict_dir, file), 'r') as f:
            predictions.extend(f.readlines())

    
    # load test data
    with open(test_data, 'r') as f:
        test_data = json.load(f)
    question2qid = {item['question'].split(' The linked')[0]: item['qid'] for item in test_data}
    # results
    results = []
    
    # execute query
    for line in tqdm(predictions):
        try:
            if type(line) == str:
                line = json.loads(line)
            final_s_exp = line['final_s_exp']
            if 'The linked' in line['question']:
                question = line['question'].split(' The linked')[0]
            else:
                question = line['question'].split('\n')[0]
            if final_s_exp is None:
                final_s_exp = 'null'
            if final_s_exp != 'null':
                try:
                    sparql = lisp_to_sparql(final_s_exp)
                    answers = execute_query(sparql)
                    answers = list(answers.values())[0]
                except:
                    answers = "null"
            else:
                answers = "null"
            results.append(json.dumps({
                'qid': question2qid[question],
                'question': question,
                'logical_form': final_s_exp,
                'answer': answers
            }))
        except Exception as e:
            print(e)
            import ipdb
            ipdb.set_trace()

    if not os.path.exists(post_processed_dir):
        os.makedirs(post_processed_dir)

    with open(f'{post_processed_dir}/processed.txt', 'w') as f:
        f.writelines("\n".join(results))