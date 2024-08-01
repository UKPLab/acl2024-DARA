import json

with open('./cvt_types.txt','r') as f:
    context = f.readlines()
context = [x.strip('\n') for x in context]

with open('./cvt_relations.json', 'r') as f:
    cvt_rels = json.load(f)

new_rels = []
for rel in cvt_rels:
    cvt_rel = False
    for type in context:
        if type in rel:
            new_rels.append(rel)
            cvt_rel = True
            break
    if not cvt_rel:
        print(rel)

with open('./cvt_relation_new.json', 'w') as f:
    json.dump(new_rels, f, indent=4)

    

    # if rel['relation'] == 'type.object.type':
    #     print(rel['entity'])
