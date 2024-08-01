from .sparql_executor import get_in_relations, get_out_relations, execute_query, get_desc_of_relation, get_friendly_name
from .logic_form_util import lisp_to_sparql
from .retriever import Retriever 
import random
import json
import ipdb
import os

class KGQuerier():
    def __init__(self, load_retriever=True) -> None:
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.bidirection_relation = self._load_bidir_relation()
        if load_retriever:
            self.retriever = Retriever()
        else:
            self.retriever = None
        with open(os.path.join(self.current_dir, "./kg_data/ontology/fb_roles"), 'r') as f:
            content = f.readlines()
        
        with open(os.path.join(self.current_dir, "./kg_data/freebase/cvt_relations.json"), 'r') as f:
            self.cvt_relations = json.load(f)

        with open(os.path.join(self.current_dir, './kg_data/freebase/rel2desc.json'), 'r') as f:
            self.rel2desc = json.load(f)

        self.range_info = {}
        for line in content:
            line = line.replace("\n", "")
            fields = line.split(" ")
            self.range_info[fields[1]] = (fields[0], fields[2])
            
    def get_classes_from_FB(self, question, var_name, query, topk=10):

        sparql = lisp_to_sparql(query, get_type=True)
        types = execute_query(sparql)
        types = list(types.values())[0]

        if len(types) > topk:
            types_ix = self.retriever.get_relevant_schemas(question, types, topk=topk)
            types = [types[ix] for ix in types_ix]
        
        return f'''{var_name} has the following classes: {", ".join(types)}.'''

    def _load_bidir_relation(self):
        with open(os.path.join(self.current_dir, 'kg_data/ontology/reverse_properties'), 'r') as f:
            reverse_properties = f.readlines()

        bidirection_relation = {}
        for line in reverse_properties:
            line = line.strip().split('\t')
            bidirection_relation[line[0]] = line[1]
        return bidirection_relation
        
    def execute_query(self, query):
        return execute_query(query)
    
    def verify_query(self, query):
        components = query.split(' ')
        operator = components[0].strip('(')
        if operator in ['JOIN', 'GE', 'GT', 'LE', 'LT']:
            if '(R' in components[1]:
                relation = components[2].strip(')')
                # head rel tail
                try:
                    type = self.range_info[relation][1]
                except:
                    type = relation.split('.')[-1]

            else:
                relation = components[1]
                try:
                    type = self.range_info[relation][0]
                except:
                    type = '.'.join(relation.split('.')[:-1])

            desc = get_desc_of_relation(type)
            if desc:
                desc = '.'.join(desc.split('.')[:2])
            else:
                desc = 'There is no additional description of this type.'
        
        else:
            return "no necessary to validate this s-expression."

        return f"This s-expression returns instances of {type}. {desc}."
        

    def schema_desc_template(self, type, relation, direction='outgoing relation'):
        relation_desc = get_desc_of_relation(relation)
        if not relation_desc:
            splitted = relation.split('.')[-2:]
            relation_desc = f"the {splitted[-1].replace('_', ' ')} of {splitted[-2].replace('_',' ')}."

        relation_desc = relation_desc[0].lower() + relation_desc[1:]
        
        type_desc = get_desc_of_relation(type)

        if not type_desc:
            type_desc = " ".join(type.split('.')[-1].split('_'))
        type_desc = '.'.join(type_desc.split('.')[:2])

        if direction == 'outgoing relation':
            rel_desc = f"the outgoing relation '{relation}', which describes {relation_desc} The type of its tail entities is '{type}' ({type_desc})."
        elif direction == 'incoming relation':
            rel_desc = f"the incoming relation '{relation}', which describes {relation_desc} The type of its head entities is '{type}' ({type_desc})."
        elif direction == 'bi relation':
            rel_desc = f"'{relation}', which describes {relation_desc} The type of its tail entities is '{type}' ({type_desc})."
        else:
            rel_desc = "None"
        return rel_desc

    def get_out_relations(self, query):
        return get_out_relations(query)

    def get_most_relevant_relations_from_FB(self, query, question, correct_rel=None, topk=5, mode='eval'):
        
        relations = self.retriever.get_relevant_relations_from_db(query, topk)
        if not correct_rel:
            try:
                correct_rel = self.q2relation[question]
                correct_rel = [rel[1] for rel in correct_rel]
            except:
                pass

        # used for constructing dataset
        if correct_rel:
            if type(correct_rel) is str:
                correct_rel = [correct_rel]
            sampled_ix = random.sample(list(range(len(relations))), len(correct_rel))
            concat_relations = "\n".join(relations)
            for ix, rel in enumerate(correct_rel):
                if rel not in concat_relations:
                    # ipdb.set_trace()
                    print('correct relation not in relations')
                    relations[sampled_ix[ix]] = self.schema_desc_template(self.range_info[rel][1], rel, direction='bi')

        # relations = [rel+' (CVT)' if rel in self.cvt_relations else rel for rel in relations]
        concat_relations = ".\n".join(relations)
        return f"The most relevant relations are {concat_relations}"

    def get_relevant_relations_from_FB(self, query, correct_rel=None, topk=5, mode='eval'):

        relations = self.retriever.get_relevant_relations_from_db(query, topk)
        relations = [rel.split(',')[0].strip("'") for rel in relations]


        if correct_rel:
            if type(correct_rel) is str:
                correct_rel = [correct_rel]
            sampled_ix = random.sample(list(range(len(relations))), len(correct_rel))
            for ix, rel in enumerate(correct_rel):
                if rel not in relations:
                    print('correct relation not in relations')
                    relations[sampled_ix[ix]] = rel
        # relations = [rel+' (CVT)' if rel in self.cvt_relations else rel for rel in relations]
        if mode == 'train':
            if len(correct_rel) > 1:
                candidate = correct_rel[:2]
            else:
                if relations[0] != correct_rel[0]:
                    candidate = correct_rel + [relations[0]]
                else:
                    candidate = correct_rel + [relations[1]]

            return "The relevant relations are " + ", ".join(relations) + ".", list(set(candidate))
     
        return "The relevant relations are " + ", ".join(relations) + "."

    def get_relevant_classes_from_FB(self, query, correct_cls=None, topk=10):
        classes = self.retriever.get_relevant_classes_from_db(query, topk)
        # used for constructing dataset
        if correct_cls:
            if type(correct_cls) is str:
                correct_cls = [correct_cls]
            sampled_ix = random.sample(list(range(len(classes))), len(correct_cls))
            for ix, cls in enumerate(correct_cls):
                if cls not in classes:
                    print('correct class not in classes')
                    classes[sampled_ix[ix]] = cls
        return "The relevant classes are " + ", ".join(classes)+"."
    
    def get_descriptions_from_FB(self, relation):
        relations = relation.split(', ')
        explanation = ""
        for ix, relation in enumerate(relations):
            rel = relation.split(' ')[0]
            try:
                direction = relation.split(' ')[1]
            except:
                direction = 'bi'
            if rel in self.rel2desc and direction != 'bi':
                if direction == '(incoming)':
                    explanation += f"{ix+1}. {self.rel2desc[rel][0]} "

                elif direction == '(outgoing)':
                    explanation += f"{ix+1}. {self.rel2desc[rel][1]} "


            else:
                if direction == '(outgoing)':
                    index = -1
                elif direction == '(incoming)':
                    index = -2
                else:
                    index = -1
                if rel not in self.range_info:
                    type = rel.split('.')[index]
                else:
                    type = self.range_info[rel][index]
                
                rel_desc = self.schema_desc_template(type, rel, direction=direction.strip('(').strip(')')+ ' relation')
                explanation += f"{ix+1}. {rel_desc} "

        return explanation
    
    def get_two_stages_relations_from_FB(self, var_name, query, question, gold_rel=None, topk=5, mode='eval'):

        if query.startswith('m.') or query.startswith('g.'):
            # get properties of entity
            in_relations = get_in_relations(query)
            out_relations = get_out_relations(query)
        else:
            outgoing_sparql = lisp_to_sparql(query, get_relation=True)
            incoming_sparql = outgoing_sparql.replace('?x ?outgoing ?obj .', '?sub ?incoming ?x .').replace('outgoing', 'incoming')
            try:
                out_relations = execute_query(outgoing_sparql)['outgoing']
                out_relations = [rel for rel in out_relations if not "http://" in rel]
            # no outgoing relations
            except KeyError:
                out_relations = []
            try:
                in_relations = execute_query(incoming_sparql)['incoming']
                in_relations = [rel for rel in in_relations if not "http://" in rel]
            except KeyError:
                in_relations = []

        new_in_relations, relations = [], []
      
        for relation in in_relations:
            if (relation in self.bidirection_relation and self.bidirection_relation[relation] in out_relations) or ('http' in relation):
                continue
            else:
                new_in_relations.append(relation)

        out_relations = [rel for rel in out_relations if 'http' not in rel]

        num_relations = len(out_relations) + len(new_in_relations)

        relations_desc, selected_out_relations, selected_in_relations = [], [], []

        for rel in out_relations:
            if rel in self.rel2desc:
                relations_desc.append(self.rel2desc[rel][1])
            else:
                if rel not in self.range_info:
                    type = rel.split('.')[-1]
                else:
                    type = self.range_info[rel][1]
                rel_desc = self.schema_desc_template(type, rel)
                relations_desc.append(rel_desc)

        for rel in new_in_relations:
            if rel in self.rel2desc:
                relations_desc.append(self.rel2desc[rel][0])
            else:
                if rel not in self.range_info:
                    type = rel.split('.')[-2]
                else:
                    type = self.range_info[rel][0]
                rel_desc = self.schema_desc_template(type, rel, direction='incoming relation')
                relations_desc.append(rel_desc)
        

        if num_relations > topk:
            
            ixs = self.retriever.get_relevant_schemas(question, relations_desc, topk=topk)
            for ix in ixs:
                if ix < len(out_relations):
                    selected_out_relations.append(out_relations[ix])
                else:
                    selected_in_relations.append(new_in_relations[ix-len(out_relations)])
        else:
            selected_out_relations = out_relations
            selected_in_relations = new_in_relations
                    
        if gold_rel:
            if gold_rel[0] == 'incoming relation' and gold_rel[1] not in selected_in_relations:
                # ipdb.set_trace()
                print(f'gold relation {gold_rel[1]} not in selected_in_relations')
                selected_in_relations = [gold_rel[1]] + selected_in_relations[:-1]

            elif gold_rel[0] == 'outgoing relation' and gold_rel[1] not in selected_out_relations:
                # ipdb.set_trace()
                print(f'gold relation {gold_rel[1]} not in selected_out_relations')
                selected_out_relations = [gold_rel[1]] + selected_out_relations[:-1]

        incoming_template = f"The incoming relations are [{', '.join(selected_in_relations)}]. "
   
        outgoing_template = f"The outgoing relations are [{', '.join(selected_out_relations)}]."

        response = f'''{var_name} has following relations. {outgoing_template} {incoming_template}'''

        if mode == 'train':
            candidates = [gold_rel[1] + f" ({gold_rel[0].split(' ')[0]})"]

            all_relations = selected_out_relations + selected_in_relations
            all_relations.remove(gold_rel[1])
            selected = random.choice(all_relations)
            if selected in selected_out_relations:
                selected += ' (outgoing)'
            else:
                selected += ' (incoming)'
            candidates.append(selected)

            random.shuffle(candidates)
            return response, candidates
        return response

    def get_top_relations_from_FB(self, var_name, query, question, gold_rel=None, topk=5, mode='eval'):

        if query.startswith('m.') or query.startswith('g.'):
            # get properties of entity
            in_relations = get_in_relations(query)
            out_relations = get_out_relations(query)
        else:
            outgoing_sparql = lisp_to_sparql(query, get_relation=True)
            incoming_sparql = outgoing_sparql.replace('?x ?outgoing ?obj .', '?sub ?incoming ?x .').replace('outgoing', 'incoming')
            try:
                out_relations = execute_query(outgoing_sparql)['outgoing']
                out_relations = [rel for rel in out_relations if not "http://" in rel]
            # no outgoing relations
            except KeyError:
                out_relations = []
            try:
                in_relations = execute_query(incoming_sparql)['incoming']
                in_relations = [rel for rel in in_relations if not "http://" in rel]
            except KeyError:
                in_relations = []
        
        new_in_relations, relations = [], []
      
        for relation in in_relations:
            if (relation in self.bidirection_relation and self.bidirection_relation[relation] in out_relations) or ('http' in relation):
                continue
            else:
                new_in_relations.append(relation)

        out_relations = [rel for rel in out_relations if 'http' not in rel]

        num_relations = len(out_relations) + len(new_in_relations)
        # relations.extend(out_relations)
        # relations.extend(new_in_relations)
        try:
            gold_relations = self.q2relation[question]
            # ipdb.set_trace()
            for rel in gold_relations:
                if rel[0]== 'incoming' and rel[1] in in_relations:
                    if rel[1] in self.bidirection_relation:
                        gold_rel = ('outgoing relation', self.bidirection_relation[rel[1]])
                    else:
                        gold_rel = ('incoming relation', rel[1])
                    break
                elif rel[0] == 'outgoing' and rel[1] in out_relations:
                    gold_rel = ('outgoing relation', rel[1])
                    break
        except:
            pass

        relations_desc, selected_relations = [], []

        for rel in out_relations:
            if rel in self.rel2desc:
                relations_desc.append(self.rel2desc[rel][1])
            else:
                if rel not in self.range_info:
                    type = rel.split('.')[-1]
                else:
                    type = self.range_info[rel][1]
                rel_desc = self.schema_desc_template(type, rel)
                relations_desc.append(rel_desc)

        for rel in new_in_relations:
            if rel in self.rel2desc:
                relations_desc.append(self.rel2desc[rel][0])
            else:
                if rel not in self.range_info:
                    type = rel.split('.')[-2]
                else:
                    type = self.range_info[rel][0]
                rel_desc = self.schema_desc_template(type, rel, direction='incoming relation')
                relations_desc.append(rel_desc)
        

        if num_relations > topk:
            
            # ipdb.set_trace()

            ixs = self.retriever.get_relevant_schemas(question, relations_desc, topk=topk)
            for ix in ixs:
                # if ix < len(out_relations):
                selected_relations.append(relations_desc[ix])
                # else:
                    # selected_relations.append(('incoming relation', relations_desc[ix]))
        else:
            selected_relations = relations_desc
                    
        if gold_rel and gold_rel[1] not in "\n".join(selected_relations):
            # ipdb.set_trace()
            print('gold relation not in selected relations')
            if gold_rel[0] == 'incoming relation':
                if gold_rel[1] not in self.range_info:
                    type = gold_rel[1].split('.')[-2]
                else:
                    type = self.range_info[gold_rel[1]][0]
            else:
                if gold_rel[1] not in self.range_info:
                    type = gold_rel[1].split('.')[-1]
                else:
                    type = self.range_info[gold_rel[1]][1]
            gold_rel_desc = self.schema_desc_template(type, rel, direction=gold_rel[0])
            selected_relations = [gold_rel_desc] + selected_relations[:-1]

        # print(selected_relations)
        concat_relations = '\n'.join(selected_relations)
        # concat_relation = '\n'.join(selected_relations)
        response = f'''The most relevant relations of {var_name} are {concat_relations}'''
        return response


    def get_relations_from_FB(self, var_name, query, question, gold_rel=None, topk=5, mode='eval'):

        if query.startswith('m.') or query.startswith('g.'):
            # get properties of entity
            in_relations = get_in_relations(query)
            out_relations = get_out_relations(query)
        else:
            outgoing_sparql = lisp_to_sparql(query, get_relation=True)
            incoming_sparql = outgoing_sparql.replace('?x ?outgoing ?obj .', '?sub ?incoming ?x .').replace('outgoing', 'incoming')
            try:
                out_relations = execute_query(outgoing_sparql)['outgoing']
                out_relations = [rel for rel in out_relations if not "http://" in rel]
            # no outgoing relations
            except KeyError:
                out_relations = []
            try:
                in_relations = execute_query(incoming_sparql)['incoming']
                in_relations = [rel for rel in in_relations if not "http://" in rel]
            except KeyError:
                in_relations = []
        
        new_in_relations, relations = [], []
      
        for relation in in_relations:
            if (relation in self.bidirection_relation and self.bidirection_relation[relation] in out_relations) or ('http' in relation) or relation in self.range_info:
                continue
            else:
                new_in_relations.append(relation)

        out_relations = [rel for rel in out_relations if 'http' not in rel and rel in self.range_info]

        num_relations = len(out_relations) + len(new_in_relations)
        relations.extend(out_relations)
        relations.extend(new_in_relations)
        try:
            gold_relations = self.q2relation[question]
            # ipdb.set_trace()
            for rel in gold_relations:
                if rel[0]== 'incoming' and rel[1] in in_relations:
                    if rel[1] in self.bidirection_relation:
                        gold_rel = ('outgoing relation', self.bidirection_relation[rel[1]])
                    else:
                        gold_rel = ('incoming relation', rel[1])
                    break
                elif rel[0] == 'outgoing' and rel[1] in out_relations:
                    gold_rel = ('outgoing relation', rel[1])
                    break
        except:
            pass

        if num_relations > topk:
            selected_in_relations, selected_out_relations = [], []
            num_out = min(len(out_relations), topk//2)
            num_in = min(len(new_in_relations), topk-num_out)
            # ipdb.set_trace()
            if gold_rel:
                selected_out_relations = out_relations[:num_out]
                selected_in_relations = new_in_relations[:num_in]
                if gold_rel[1] not in selected_in_relations and gold_rel[1] not in selected_out_relations:
                    if gold_rel[0] == 'incoming relation':
                        selected_in_relations = list([gold_rel[1]] + selected_in_relations[:-1])
                    elif gold_rel[0] == 'outgoing relation':
                        selected_out_relations = list([gold_rel[1]] + selected_out_relations[:-1])
                
                random.shuffle(selected_in_relations)
                random.shuffle(selected_out_relations)

            else:
                relations_desc = []
                for rel in out_relations:
                    try:
                        type = self.range_info[rel][1]
                    except:
                        continue
                    type_desc = get_desc_of_relation(type)
                    if not type_desc:
                        type_desc = " ".join(type.split('.')[-1].split('_'))
                    relations_desc.append(type + ': ' + type_desc)
                
                for rel in new_in_relations:
                    try:
                        type = self.range_info[rel][0]
                    except:
                        continue
                    type_desc = get_desc_of_relation(type)
                    if not type_desc:
                        type_desc = " ".join(type.split('.')[-1].split('_'))
                    relations_desc.append(type + ': ' + type_desc)
                ixs = self.retriever.get_relevant_schemas(question, relations_desc, topk=topk)
                for ix in ixs:
                    if ix < len(out_relations):
                        selected_out_relations.append(out_relations[ix])
                    else:
                        selected_in_relations.append(new_in_relations[ix-len(out_relations)])
                    
                        
            new_in_relations, out_relations = selected_in_relations, selected_out_relations

        if gold_rel[0] == 'incoming relation' and gold_rel[1] not in new_in_relations:
            print(f'gold relation {gold_rel[1]} not in new_in_relations')
            print(f'the question is {question}')
        if gold_rel[0] == 'outgoing relation' and gold_rel[1] not in out_relations:
            print(f'gold relation {gold_rel[1]} not in out_relations')
            print(f'the question is {question}')

        new_in_relations = [rel +' (CVT)' if rel in self.cvt_relations else rel for rel in new_in_relations]
        out_relations = [rel+' (CVT)' if rel in self.cvt_relations else rel for rel in out_relations]


        incoming_template = f"The incoming relations are [{', '.join(new_in_relations)}]. "
   
        outgoing_template = f"The outgoing relations are [{', '.join(out_relations)}]."

        response = f'''{var_name} has following relations. {outgoing_template} {incoming_template}'''
        if mode == 'train':
            # import ipdb
            # ipdb.set_trace()

            all_relations = new_in_relations + out_relations

            gold_relation = gold_rel[1]
            try:
                if gold_rel[1] not in all_relations:
                    gold_relation += ' (CVT)'
                all_relations.remove(gold_relation)
            except:
                ipdb.set_trace()
        
            candidates = self.retriever.get_relevant_schemas(gold_rel[1], all_relations, threshold=0.75)
            candidates += [gold_rel[1]]
            # tokenized_relations = [rel.split('.') for rel in all_relations]
            # bm25 = BM25Okapi(tokenized_relations)
            # candidate_relations = bm25.get_top_n(gold_rel[1].split('.'), tokenized_relations, n=3)
            # candidate_relations = ['.'.join(rel) for rel in candidate_relations]
            random.shuffle(candidates)

            return response, candidates
        else:
            return response


if __name__ == '__main__':
    querier = KGQuerier(load_retriever=False)
    # # print(querier.get_classes_from_FB('s-exp-1', '(AND (JOIN astronomy.celestial_object_category.subcategory_of m.04gs7c0) (JOIN astronomy.celestial_object_category.subcategories m.08n_lz))', '(JOIN film.film.soundtrack (JOIN music.soundtrack.film m.0b86qd))'))
    
    # print(querier.get_two_stages_relations_from_FB('s-exp-1', 'm.032gdy', 'where was the brickyard 400 held? ', gold_rel = None, topk=5))
    print(querier.get_descriptions_from_FB('government.politician.election_campaigns'))
    



    
