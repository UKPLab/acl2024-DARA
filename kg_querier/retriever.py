from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import pickle
import ipdb
from .sparql_executor import get_desc_of_relation
from tqdm import tqdm

class Retriever:
    def __init__(self) -> None:
        self.current_dir = os.path.dirname(__file__)
        # self.bi_model = SentenceTransformer(os.path.join(self.current_dir, 'retriever'))
        self.bi_model = SentenceTransformer('Haixx/relation_retriever')
        self.load_relation_embedding()
        self.load_class_embedding()

    def load_relation_embedding(self):

        rel_db_path = os.path.join(self.current_dir, 'kg_data/rel_desc_db_embed_custom.pkl')
        if os.path.exists(rel_db_path):
            print(f'loading relation desc embedding from {rel_db_path}')
            # start1 = time()
            with open(rel_db_path, 'rb') as f:
                candidates = pickle.load(f)
            # end1 = time()
            # print('time', end1-start1)
            self.candidate_rel, self.candidate_rel_embed = candidates['relations'], candidates['embeddings']

        else:
            print('saving relation embedding into file')
            with open(os.path.join(self.current_dir, 'kg_data/ontology/fb_roles'),'r', encoding='utf8') as f:
                content = f.readlines()

            self.candidate_rel = []
            for ix, line in tqdm(enumerate(content)):
                triplets = line.split(' ')
                if len(triplets) != 3:
                    print(triplets)
                relation = triplets[1]
                relation_desc = get_desc_of_relation(relation)
                if not relation_desc:
                    splitted = relation.split('.')[-2:]
                    try:
                        relation_desc = f"the {splitted[-1].replace('_', ' ')} of {splitted[-2].replace('_',' ')}."
                    except:
                        ipdb.set_trace()
                        continue

                relation_desc = relation_desc[0].lower() + relation_desc[1:]

                # get desc of relations
                incoming_type = triplets[0]
                outgoing_type = triplets[2].strip('\n')

                out_type_desc = get_desc_of_relation(outgoing_type)
                if not out_type_desc:
                    out_type_desc = " ".join(outgoing_type.split('.')[-1].split('_'))
                out_type_desc = '.'.join(out_type_desc.split('.')[:2])

                rel_desc = f"'{relation}', which describes {relation_desc} Its type is '{outgoing_type}' ({out_type_desc})"
                self.candidate_rel.append(rel_desc)

            # self.candidate_rel = list(set(candidate_rel))
            self.candidate_rel_embed = self.bi_model.encode(self.candidate_rel)
            candidates = {'relations': self.candidate_rel, 'embeddings': self.candidate_rel_embed}
            with open(rel_db_path, 'wb') as f:
                pickle.dump(candidates,f)

    def load_class_embedding(self):
        class_db_path = os.path.join(self.current_dir, 'kg_data/class_db_embed_custom.pkl')
        if os.path.exists(class_db_path):
            print(f'loading relation embedding from {class_db_path}')
            # start1 = time()
            with open(class_db_path, 'rb') as f:
                candidates = pickle.load(f)
            # end1 = time()
            # print('time', end1-start1)
            self.candidate_cls, self.candidate_cls_embed = candidates['classes'], candidates['embeddings']

        else:
            print('saving classes embedding into file')
            with open(os.path.join(self.current_dir, 'kg_data/ontology/fb_types','r', encoding='utf8')) as f:
                content = f.readlines()
            candidate_cls = []
            for line in content:
                triplets = line.strip().split(' ')
                # print(triplets)
                triplets = [trip for trip in triplets if trip != '.']
                # print('triplets',  triplets)
                # exit()
                if len(triplets) != 3:
                    print('line', line)
                    exit()
                cls = triplets[0]
                if cls != 'common.topic':
                    candidate_cls.append(cls)
            self.candidate_cls = list(set(candidate_cls))
            self.candidate_cls_embed = self.bi_model.encode(self.candidate_cls)
            candidates = {'classes': self.candidate_cls, 'embeddings': self.candidate_cls_embed}
            with open(class_db_path, 'wb') as f:
                pickle.dump(candidates,f)

    def get_relevant_schemas(self, query, candidates, topk=5, threshold=0):

        candidates_embed = self.bi_model.encode(candidates)
        query_embed = self.bi_model.encode(query)
        scores = util.cos_sim(query_embed, candidates_embed)
        scores = scores.cpu().detach().numpy()
        
        if threshold > 0:
            try:
            # get index where sim larger than threshold index

                rel_indices = np.argwhere(scores >= threshold)[:,1][:2]
                if rel_indices.all():
                    return [candidates[ix] for ix in rel_indices]
                else:
                    return []
            except Exception as e:
                ipdb.set_trace()
                print('the error is', e)

        elif topk > 0:
            rel_indices = np.argsort(scores)[:, ::-1][:,:topk]
            return rel_indices[0]

    def get_relevant_classes_from_db(self, query, topk=10):
        # print('classes', self.candidate_cls)
        query_embed = self.bi_model.encode(query)
        sims = util.cos_sim(query_embed, self.candidate_cls_embed)
        sims = sims.cpu().detach().numpy()
        cls_indices = np.argsort(sims)[:, ::-1][:,:topk]
        relevant_classes = []
        for ix in cls_indices[0]:
            relevant_classes.append(self.candidate_cls[ix])
        return relevant_classes


    def get_relevant_relations_from_db(self, query, topk=15):
        
        query_embed = self.bi_model.encode(query)
        
        sims = util.cos_sim(query_embed, self.candidate_rel_embed)
        sims = sims.cpu().detach().numpy()
        
        rel_indices = np.argsort(sims)[:, ::-1][:,:topk]
        # import ipdb
        # ipdb.set_trace()
        relevant_relations = []
        for ix in rel_indices[0]:
            relevant_relations.append(self.candidate_rel[ix])
        return relevant_relations

if __name__ == '__main__':
    # model_path = ''
    retriever = Retriever()
    # retriever
    task = ['Step 2: find the end date of each conference', 'image.png']
    for t in task:
        res = retriever.get_relevant_relation_from_db(t)
        print(res)
        exit()