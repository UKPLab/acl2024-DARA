import json
import argparse
import networkx as nx
from collections import defaultdict
from typing import List, Set, Dict
# from .error_anlaysis import post_process_predictions_react
from tqdm import tqdm
from kg_querier.sparql_executor import execute_query
from kg_querier.logic_form_util import lisp_to_sparql
import os
import re
from utils.data_utils import post_process_predictions_react


function_map = {'le': '<=', 'ge': '>=', 'lt': '<', 'gt': '>'}

def process_ontology(fb_roles_file, fb_types_file, reverse_properties_file):
    reverse_properties = {}
    with open(reverse_properties_file, 'r') as f:
        for line in f:
            reverse_properties[line.split('\t')[0]] = line.split('\t')[1].replace('\n', '')

    with open(fb_roles_file, 'r') as f:
        content = f.readlines()

    relation_dr = {}
    relations = set()
    for line in content:
        fields = line.split()
        relation_dr[fields[1]] = (fields[0], fields[2])
        relations.add(fields[1])

    with open(fb_types_file, 'r') as f:
        content = f.readlines()

    upper_types = defaultdict(lambda: set())

    types = set()
    for line in content:
        fields = line.split()
        upper_types[fields[0]].add(fields[2])
        types.add(fields[0])
        types.add(fields[2])

    return reverse_properties, relation_dr, relations, upper_types, types


def lisp_to_nested_expression(lisp_string: str) -> List:
    """
    Takes a logical form as a lisp string and returns a nested list representation of the lisp.
    For example, "(count (division first))" would get mapped to ['count', ['division', 'first']].
    """
    stack: List = []
    current_expression: List = []
    tokens = lisp_string.split()
    for token in tokens:
        while token[0] == '(':
            nested_expression: List = []
            current_expression.append(nested_expression)
            stack.append(current_expression)
            current_expression = nested_expression
            token = token[1:]
        current_expression.append(token.replace(')', ''))
        while token[-1] == ')':
            current_expression = stack.pop()
            token = token[:-1]
    return current_expression[0]


class SemanticMatcher:
    def __init__(self, reverse_properties, relation_dr, relations, upper_types, types):
        self.reverse_properties = reverse_properties
        self.relation_dr = relation_dr
        self.relations = relations
        self.upper_types = upper_types
        self.types = types

    def same_logical_form(self, form1, form2):
        if form1.__contains__("@@UNKNOWN@@") or form2.__contains__("@@UNKNOWN@@"):
            return False
        try:
            G1 = self.logical_form_to_graph(lisp_to_nested_expression(form1))
        except Exception:
            return False
        try:
            G2 = self.logical_form_to_graph(lisp_to_nested_expression(form2))
        except Exception:
            return False

        def node_match(n1, n2):
            if n1['id'] == n2['id'] and n1['type'] == n2['type']:
                func1 = n1.pop('function', 'none')
                func2 = n2.pop('function', 'none')
                tc1 = n1.pop('tc', 'none')
                tc2 = n2.pop('tc', 'none')

                if func1 == func2 and tc1 == tc2:
                    return True
                else:
                    return False
                # if 'function' in n1 and 'function' in n2 and n1['function'] == n2['function']:
                #     return True
                # elif 'function' not in n1 and 'function' not in n2:
                #     return True
                # else:
                #     return False
            else:
                return False

        def multi_edge_match(e1, e2):
            if len(e1) != len(e2):
                return False
            values1 = []
            values2 = []
            for v in e1.values():
                values1.append(v['relation'])
            for v in e2.values():
                values2.append(v['relation'])
            return sorted(values1) == sorted(values2)

        return nx.is_isomorphic(G1, G2, node_match=node_match, edge_match=multi_edge_match)

    def get_symbol_type(self, symbol: str) -> int:
        if symbol.__contains__('^^'):   # literals are expected to be appended with data types
            return 2
        elif symbol in self.types:
            return 3
        elif symbol in self.relations:
            return 4
        else:
            return 1

    def logical_form_to_graph(self, expression: List) -> nx.MultiGraph:
        # TODO: merge two entity node with same id. But there is no such need for
        # the second version of graphquestions
        G = self._get_graph(expression)
        G.nodes[len(G.nodes())]['question_node'] = 1
        return G

    def _get_graph(self, expression: List) -> nx.MultiGraph:  # The id of question node is always the same as the size of the graph
        if isinstance(expression, str):
            G = nx.MultiDiGraph()
            if self.get_symbol_type(expression) == 1:
                G.add_node(1, id=expression, type='entity')
            elif self.get_symbol_type(expression) == 2:
                G.add_node(1, id=expression, type='literal')
            elif self.get_symbol_type(expression) == 3:
                G.add_node(1, id=expression, type='class')
                # G.add_node(1, id="common.topic", type='class')
            elif self.get_symbol_type(expression) == 4:  # relation or attribute
                domain, rang = self.relation_dr[expression]
                G.add_node(1, id=rang, type='class')  # if it's an attribute, the type will be changed to literal in arg
                G.add_node(2, id=domain, type='class')
                G.add_edge(2, 1, relation=expression)

                if expression in self.reverse_properties:   # take care of reverse properties
                    G.add_edge(1, 2, relation=self.reverse_properties[expression])

            return G

        if expression[0] == 'R':
            G = self._get_graph(expression[1])
            size = len(G.nodes())
            mapping = {}
            for n in G.nodes():
                mapping[n] = size - n + 1
            G = nx.relabel_nodes(G, mapping)
            return G

        elif expression[0] in ['JOIN', 'le', 'ge', 'lt', 'gt']:
            G1 = self._get_graph(expression=expression[1])
            G2 = self._get_graph(expression=expression[2])
            size = len(G2.nodes())
            qn_id = size
            if G1.nodes[1]['type'] == G2.nodes[qn_id]['type'] == 'class':
                if G2.nodes[qn_id]['id'] in self.upper_types[G1.nodes[1]['id']]:
                    G2.nodes[qn_id]['id'] = G1.nodes[1]['id']
                # G2.nodes[qn_id]['id'] = G1.nodes[1]['id']
            mapping = {}
            for n in G1.nodes():
                mapping[n] = n + size - 1
            G1 = nx.relabel_nodes(G1, mapping)
            G = nx.compose(G1, G2)

            if expression[0] != 'JOIN':
                G.nodes[1]['function'] = function_map[expression[0]]

            return G

        elif expression[0] == 'AND':
            G1 = self._get_graph(expression[1])
            G2 = self._get_graph(expression[2])

            size1 = len(G1.nodes())
            size2 = len(G2.nodes())
            if G1.nodes[size1]['type'] == G2.nodes[size2]['type'] == 'class':
                G2.nodes[size2]['id'] = G1.nodes[size1]['id']
                # IIRC, in nx.compose, for the same node, its information can be overwritten by its info in the second graph
                # So here for the AND function we force it to choose the type explicitly provided in the logical form
            mapping = {}
            for n in G1.nodes():
                mapping[n] = n + size2 - 1
            G1 = nx.relabel_nodes(G1, mapping)
            G2 = nx.relabel_nodes(G2, {size2: size1 + size2 - 1})
            G = nx.compose(G1, G2)

            return G

        elif expression[0] == 'COUNT':
            G = self._get_graph(expression[1])
            size = len(G.nodes())
            G.nodes[size]['function'] = 'count'

            return G

        elif expression[0].__contains__('ARG'):
            G1 = self._get_graph(expression[1])
            size1 = len(G1.nodes())
            G2 = self._get_graph(expression[2])
            size2 = len(G2.nodes())
            # G2.nodes[1]['class'] = G2.nodes[1]['id']   # not sure whether this is needed for sparql
            G2.nodes[1]['id'] = 0
            G2.nodes[1]['type'] = 'literal'
            G2.nodes[1]['function'] = expression[0].lower()
            if G1.nodes[size1]['type'] == G2.nodes[size2]['type'] == 'class':
                G2.nodes[size2]['id'] = G1.nodes[size1]['id']

            mapping = {}
            for n in G1.nodes():
                mapping[n] = n + size2 - 1
            G1 = nx.relabel_nodes(G1, mapping)
            G2 = nx.relabel_nodes(G2, {size2: size1 + size2 - 1})
            G = nx.compose(G1, G2)

            return G

        elif expression[0] == 'TC':
            G = self._get_graph(expression[1])
            size = len(G.nodes())
            G.nodes[size]['tc'] = (expression[2], expression[3])

            return G


def get_predictions(test_data, raw_predictions, reverse_properties_dict):
    if 'question' in test_data[0]:
        test_data = {entry['question'].split('The linked')[0].strip(): entry for entry in test_data}
    else:
        test_data = {entry['input']: entry for entry in test_data}
    predictions = {}
    for entry in tqdm(raw_predictions):
        entry = json.loads(entry)
        pred_question = entry['question'].split('The linked')[0].strip()
        if pred_question not in test_data: continue
        qid = test_data[pred_question]['qid']
        qid = str(qid)
        if qid not in predictions:
            predictions[qid] = []
        
        if 'failed' in entry['exec_result']:
            predictions[qid].append({'question':entry['question'],'logical_form': 'null', 'answer': 'null'})
            continue
        try:
            final_s_exp = entry['s_expression']['Final s-exp'].split('\n')[0]

        except Exception as e:
            print(e)
            print(entry['s_expression'])
            predictions[qid].append({'question':entry['question'], 'logical_form': 'null', 'answer': 'null'})
            continue
        
        final_s_exp = post_process_predictions_react(final_s_exp, entry['s_expression'], reverse_properties_dict)
        try:
            sparql = lisp_to_sparql(final_s_exp)
            answers = execute_query(sparql)
            answers = list(answers.values())[0]
        except:
            answers = "null"
        
        predictions[qid].append({'question': entry['question'], 'logical_form': final_s_exp, 'answer': answers})
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_data_path', type=str, help='The path to dataset file for evaluation (e.g., dev.json or test.json)',
                        default='')
    parser.add_argument('--predict_data_dir', type=str, default='./outputs/grailqa', help='The dir to predictions')
    parser.add_argument('--metric_output_path', type=str, default='./eval/dara/', help='The metric output path')
    
    parser.add_argument('--fb_roles', type=str, default='./kg_querier/kg_data/ontology/fb_roles', help='The path to ontology file')
    parser.add_argument('--fb_types', type=str, default='./kg_querier/kg_data/ontology/fb_types', help='The path to ontology file')
    parser.add_argument('--reverse_properties', type=str, default='./kg_querier/kg_data/ontology/reverse_properties',
                        help='The path to ontology file')

    args = parser.parse_args()

    reverse_properties, relation_dr, relations, upper_types, types = process_ontology(args.fb_roles, args.fb_types,
                                                                                      args.reverse_properties)
    
    with open(args.gold_data_path) as f:
        gold_data = json.load(f)

    if not os.path.exists(args.metric_output_path):
        os.makedirs(args.metric_output_path)
    if not os.path.exists(f'{args.metric_output_path}/predict.json'):
        raw_pred_data_list = os.listdir(args.predict_data_dir)

        pred_result = []
        for file in raw_pred_data_list:
            with open(os.path.join(args.predict_data_dir, file)) as f:
                pred_output = f.readlines()
            pred_result.extend(pred_output)
        if 'answer' not in json.loads(pred_result[0]):
            predict = get_predictions(gold_data, pred_result, reverse_properties)
        else: 
            predict = {}
            for entry in pred_result:
                entry = json.loads(entry)
                predict[str(entry['qid'])] = [entry]
        
        with open(f'{args.metric_output_path}/predict.json', 'w') as f:
            json.dump(predict, f, ensure_ascii=False, indent=4)

    else:    
        with open(f'{args.metric_output_path}/predict.json','r') as f:
            predict = json.load(f)  # should be of format {qid: {logical_form: <str>, answer: <list>}}

    # if 'answer' in gold_data[0]:
    matcher = SemanticMatcher(reverse_properties, relation_dr, relations, upper_types, types)

    em_sum, f1_sum = 0, 0
    level_count = defaultdict(lambda : 0)
    level_em_sum = defaultdict(lambda : 0)
    level_f1_sum = defaultdict(lambda : 0)
    errors = []
    mid_extractor = re.compile(r'm\.[0-9a-zA-Z_]+')
    linked_error, reasoning_error, relation_error = 0, 0, 0
    cnt = 0
    no_gold_answer = 0
    linked_error_mid = []

    # no_s_exp = 0
    for item in gold_data:
        linked_correct = False
        if item['qid'] == 'null': continue
        # try:
        #     s_exp_mid = mid_extractor.findall(item['question'])
        # except:
        #     # no_s_exp += 1
        #     continue
        if str(item['qid']) not in predict:
            print(f"no predctions of {item['qid']}")
            continue
        for cand in predict[str(item['qid'])]:
            cnt += 1
            selected_cand = cand

        # fake
        item['level'] = 'iid'
        level_count[item['level']] += 1
        # for webqsp
        answer = set()
        if 'answer' in item:
            if item['answer'] != 'null':
                for a in item['answer']:
                    answer.add(a['answer_argument'])
            else:
                no_gold_answer += 1
                continue
                
        # ipdb.set_trace()

        if item['s_expression'] is not None:
            em = matcher.same_logical_form(selected_cand['logical_form'], item['s_expression'])
            em_sum += em
        else:
            em = 0
        level_em_sum[item['level']] += em
        if em:
            f1_sum += 1
            level_f1_sum[item['level']] += 1
        else:
            pred_dic = {}
            predict_answer = set(selected_cand['answer'])     
            if len(predict_answer.intersection(answer)) != 0:
                precision = len(predict_answer.intersection(answer)) / len(predict_answer)
                recall = len(predict_answer.intersection(answer)) / len(answer)
                f1 = (2 * recall * precision / (recall + precision))
                f1_sum += f1
                if f1 != 1:
                    if item['s_expression'] is None:
                        error_type = 'no_gold_s_expression'
                    elif item['s_expression'].count("JOIN") != selected_cand['logical_form'].count("JOIN"):
                        reasoning_error += 1
                        error_type = 'reasoning_error'
                    else:
                        relation_error += 1
                        error_type = 'relation_error'
                    errors.append({'qid':item['qid'],'question':item['question'], 'pred': selected_cand['logical_form'], 's_expression':item['s_expression'], 'error_type': error_type})

                level_f1_sum[item['level']] += (2 * recall * precision / (recall + precision))
            
            else:
                if item['s_expression'] is None:
                    error_type = 'no_gold_s_expression'

                elif item['s_expression'].count("JOIN") != selected_cand['logical_form'].count("JOIN"):
                        reasoning_error += 1
                        error_type = 'reasoning_error'
                else:
                    relation_error += 1
                    error_type = 'relation_error'

                errors.append({'qid':item['qid'],'question':item['question'], 'pred': selected_cand['logical_form'], 's_expression':item['s_expression'], 'error_type': error_type})

    stats = {}
    print(cnt)
    print(em_sum, f1_sum)
    print('no gold answer', no_gold_answer)
    stats['em'] = em_sum / (len(gold_data))
    stats['f1'] = f1_sum / (len(gold_data))
    stats['linked_error'] = linked_error
    stats['reasoning_error'] = reasoning_error
    stats['relation_error'] = relation_error
    stats['fix_f1'] = f1_sum / (len(gold_data) - linked_error)
    stats['total_num'] = len(gold_data)
    stats['f1_sum'] = f1_sum
    print(stats)
    json.dump(stats, open(f"{args.metric_output_path}/metrics.json", 'w'))
    json.dump(errors, open(f"{args.metric_output_path}/errors.json", 'w'), indent=4, ensure_ascii=False)
