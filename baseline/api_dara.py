import json
import re
from typing import Union, List
from pathlib import Path
from kg_querier.logic_form_util import lisp_to_nested_expression, lisp_to_sparql
from kg_querier.retriever import Retriever
from kg_querier.sparql_executor import get_desc_of_relation
import os

root = str(Path(__file__).parent.absolute())

with open(os.path.join(root, 'kg_querier', "kg_data/ontology/vocab.json")) as f:
    vocab = json.load(f)
    attributes = vocab["attributes"]
    relations = vocab["relations"]

range_info = {}


with open(os.path.join(root, 'kg_querier', "kg_data/ontology/fb_roles"), 'r') as f:
    for line in f:
        line = line.replace("\n", "")
        fields = line.split(" ")
        range_info[fields[1]] = (fields[0], fields[2])


class_info = {}
with open(os.path.join(root, 'kg_querier', "kg_data/ontology/fb_types"), 'r') as f:
    for line in f:
        line = line.replace("\n", "")
        fields = line.split(" ")
        if fields[0] not in class_info:
            class_info[fields[0]] = []
        class_info[fields[0]].append(fields[2])

with open(os.path.join(root, 'kg_querier', 'kg_data/freebase/rel2desc.json'), 'r') as f:
    rel2desc = json.load(f)


retriever = Retriever()

variable_relations_cache = {}
variable_attributes_cache = {}

def binary_nesting(function: str, elements: List[str], types_along_path=None) -> str:
    if len(elements) < 2:
        print("error: binary function should have 2 parameters!")
    if not types_along_path:
        if len(elements) == 2:
            return '(' + function + ' ' + elements[0] + ' ' + elements[1] + ')'
        else:
            return '(' + function + ' ' + elements[0] + ' ' + binary_nesting(function, elements[1:]) + ')'
    else:
        if len(elements) == 2:
            return '(' + function + ' ' + types_along_path[0] + ' ' + elements[0] + ' ' + elements[1] + ')'
        else:
            return '(' + function + ' ' + types_along_path[0] + ' ' + elements[0] + ' ' \
                   + binary_nesting(function, elements[1:], types_along_path[1:]) + ')'

def expression_to_lisp(expression) -> str:
    rtn = '('
    for i, e in enumerate(expression):
        if isinstance(e, list):
            rtn += expression_to_lisp(e)
        else:
            rtn += e
        if i != len(expression) - 1:
            rtn += ' '

    rtn += ')'
    return rtn

def postprocess_raw_code(raw_lisp):
    expression = lisp_to_nested_expression(raw_lisp)
    if expression[0] in ["ARGMAX", "ARGMIN"] and len(expression) > 3:
        expression[2] = binary_nesting("JOIN", expression[2:])
        expression = expression[:3]
        raw_lisp = expression_to_lisp(expression)

    splits = raw_lisp.split(' ')
    for i, s in enumerate(splits):
        if len(s) > 4 and s[-4:] == '_inv':
            splits[i] = f'(R {s[:-4]})'
        if len(s) > 5 and s[ -5:] == '_inv)':
            splits[i] = f'(R {s[:-5]}))'
    processed_lisp = ' '.join(splits)

    return processed_lisp

class Variable:
    def __init__(self, type, program):
        self.type = type
        self.program = program

    def __hash__(self) -> int:
        return hash(self.program)
    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Variable):
            return self.program == __value.program
        else:
            return False
    def __repr__(self) -> str:
        return self.program

def final_execute(variable: Variable, sparql_executor):
    program = variable.program
    processed_code = postprocess_raw_code(program)
    sparql_query = lisp_to_sparql(processed_code)

    results = sparql_executor.execute_query(sparql_query)

    return results


def get_relations(variable: Union[Variable, str], sparql_executor):
    """
    Get all relations of a variable
    :param variable: here a variable is represented as its program derivation
    :return: a list of relations
    """
    if not isinstance(variable, Variable):
        if not re.match(r'^(m|f|g)\.[\w_]+$', variable):
            raise ValueError("get_relations: variable must be a variable or an entity")

    if isinstance(variable, Variable):
        program = variable.program
        
        processed_code = postprocess_raw_code(program)
        sparql_query = lisp_to_sparql(processed_code)
        clauses = sparql_query.split("\n")
        
        new_clauses = [clauses[0], "SELECT DISTINCT ?rel\nWHERE {\n?x ?rel ?obj .\n{"]
        new_clauses.extend(clauses[1:])
        new_clauses.append("}\n}")
        new_query = '\n'.join(new_clauses)
        try:
            raw_out_relations = sparql_executor.execute_query(new_query)['rel']
        except KeyError:
            raw_out_relations = []
        
    else: # variable is an entity
        raw_out_relations = sparql_executor.get_out_relations(variable)

    raw_out_relations = list(set(raw_out_relations).intersection(set(relations)))

    # if len(raw_out_relations) > 10:
    #     out_relations = raw_out_relations[:10]
    #     if gold_relation:
    #         for rel in gold_relation:
    #             if rel not in out_relations and rel in raw_out_relations:
    #                 out_relations.insert(0, rel)
        

    # new_clauses = [clauses[0], "SELECT DISTINCT ?rel\nWHERE {\n?sub ?rel ?x .\n{"]
    # new_clauses.extend(clauses[1:])
    # new_clauses.append("}\n}")
    # new_query = '\n'.join(new_clauses)
    # in_relations = execute_query(new_query)

    rtn_str = f": [{', '.join(raw_out_relations)}]\n"
    variable_relations_cache[variable] = raw_out_relations

    return None, rtn_str

def get_classes(variable:Variable, sparql_executor):
    if not isinstance(variable, Variable):
        raise ValueError("get_classes: variable must be a variable")
    program = variable.program
    processed_code = postprocess_raw_code(program)
    sparql = lisp_to_sparql(processed_code, get_type=True)
    # print(sparql)
    types = sparql_executor.execute_query(sparql)
    types = list(types.values())[0]

    # if len(types) > topk:
    #     types_ix = retriever.get_relevant_schemas(question, types, topk=topk)
    #     types = [types[ix] for ix in types_ix]
    # print('types',types)
    # print(types)
    return None, f''': {", ".join(types)}.'''

def get_relevant_relations(task: str, sparql_executor):
    relations = retriever.get_relevant_relations_from_db(task, topk=5)
    relations = [rel.split(',')[0].strip("'") for rel in relations]
    return None, ": The relevant relations are " + ", ".join(relations) + "."

def get_relevant_attributes(task: str, sparql_executor, topk=5):
    relations = retriever.get_relevant_relations_from_db(task, topk)
    relations = [rel.split(',')[0].strip("'") for rel in relations]
    return None, ": The relevant attributes are " + ", ".join(relations) + "."

def get_relevant_classes(task:str, sparql_executor, topk=5):
    classes = retriever.get_relevant_classes_from_db(task, topk)
    return None, ": The relevant classes are " + ", ".join(classes)+"."

def get_neighbors(variable: Union[Variable, str], relation: str, sparql_executor):  # will create a new variable
    """
    Get all neighbors of a variable
    :param variable: a variable, here a variable is represented as its program derivation
    :param relation: a relation
    :return: a list of neighbors
    """
    if not isinstance(variable, Variable):
        if not re.match(r'^(m|f|g)\.[\w_]+$', variable):
            raise ValueError("get_neighbors: variable must be a variable or an entity")
    if variable not in variable_relations_cache:
        raise ValueError("You haven't called get_relations for this variable yet.")
    if not relation in variable_relations_cache[variable] and not relation in variable_attributes_cache[variable]:
        raise ValueError("get_neighbors: relation must be a relation or an attribute of the variable")
        

    rtn_str = f": variable ##, which are instances of {range_info[relation][1]}\n"

    new_variable = Variable(range_info[relation][1], 
                            f"(JOIN {relation + '_inv'} {variable.program if isinstance(variable, Variable) else variable})")

    return new_variable, rtn_str


def intersection(variable1: Variable, variable2: Variable, sparql_executor):  # will create a new variable
    """
    Get the intersection of two variables
    :param variable1: a variable
    :param variable2: a variable
    :return: a list of intersection
    """
    if variable1.type != variable2.type and (variable2.type not in class_info[variable1.type] and variable1.type not in class_info[variable2.type]):
        raise ValueError("intersection: two variables must have the same type")

    if not isinstance(variable1, Variable) or not isinstance(variable2, Variable):
        raise ValueError("intersection: variable must be a variable")

    rtn_str = f": variable ##, which are instances of {variable1.type}\n"
    new_variable = Variable(variable1.type, f"(AND {variable1.program} {variable2.program})")
    return new_variable, rtn_str


def union(variable1: set, variable2: set, sparql_executor): # will create a new variable
    """
    Get the union of two variables
    :param variable1: a variable
    :param variable2: a variable
    :return: a list of union
    """
    if variable1.type != variable2.type:
        raise ValueError("union: two variables must have the same type")

    if not isinstance(variable1, Variable) or not isinstance(variable2, Variable):
        raise ValueError("union: variable must be a variable")

    rtn_str = f": variable ##, which are instances of {variable1.type}\n"
    new_variable = Variable(variable1.type, f"(OR {variable1.program} {variable2.program})")
    return new_variable, rtn_str


def count(variable: Variable, sparql_executor):
    """
    Count the number of a variable
    :param variable: a variable
    :return: the number of a variable
    """
    rtn_str = f": variable ##, which is a number\n"
    new_variable = Variable("type.int", f"(COUNT {variable.program})")
    return new_variable, rtn_str


def get_attributes(variable: Variable, sparql_executor):
    if isinstance(variable, Variable):
        program = variable.program
        processed_code = postprocess_raw_code(program)
        sparql_query = lisp_to_sparql(processed_code)
        clauses = sparql_query.split("\n")
    
        new_clauses = [clauses[0], "SELECT DISTINCT ?rel\nWHERE {\n?x ?rel ?obj .\n{"]
        new_clauses.extend(clauses[1:])
        new_clauses.append("}\n}")
        new_query = '\n'.join(new_clauses)
        try:
            out_relations = sparql_executor.execute_query(new_query)['rel']
        except KeyError:
            out_relations = []
    else:
        out_relations = sparql_executor.get_out_relations(variable)

    out_relations = list(set(out_relations).intersection(set(attributes)))
    variable_attributes_cache[variable] = out_relations

    rtn_str = f": [{', '.join(out_relations)}]\n"

    return None, rtn_str



def argmax(variable: str, attribute: str, sparql_executor):
    """
    Get the argmax of a variable
    :param variable: a variable
    :param relation: a relation
    :return: the argmax of a variable
    """
    # program = f"(ARGMAX {variable} {attribute})"
    # processed_code = postprocess_raw_code(program)
    # sparql_query = lisp_to_sparql(processed_code)
    # answers = execute_query(sparql_query)
    if attribute not in variable_attributes_cache[variable]:
        raise ValueError("argmax: attribute must be an attribute of the variable")
    
    rtn_str = f": variable ##, which are instances of {variable.type}\n"
    new_variable = Variable(variable.type, f"(ARGMAX {variable.program} {attribute})")
    return new_variable, rtn_str

def schema_desc_template(type, relation, direction='outgoing relation'):
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
        rel_desc = f"the outgoing relation '{relation}', which describes {relation_desc} The type of its tail entity is '{type}' ({type_desc})."
    elif direction == 'incoming relation':
        rel_desc = f"the incoming relation '{relation}', which describes {relation_desc} The type of its head entity is '{type}' ({type_desc})."
    elif direction == 'bi relation':
        rel_desc = f"'{relation}', which describes {relation_desc} The type of its tail entity is '{type}' ({type_desc})."
    else:
        rel_desc = "None"
        # ipdb.set_trace()
    return rel_desc

def get_descriptions(relation:str, sparql_executor):
    relations = relation.split(', ')
    explanation = ""
    for ix, relation in enumerate(relations):
        rel = relation.split(' ')[0]
        try:
            direction = relation.split(' ')[1]
        except:
            direction = 'bi'
        if rel in rel2desc and direction != 'bi':
            if direction == '(incoming)':
                explanation += f"{ix+1}. {rel2desc[rel][0]} "

            elif direction == '(outgoing)':
                explanation += f"{ix+1}. {rel2desc[rel][1]} "
        else:
            if direction == '(outgoing)':
                index = -1
            elif direction == '(incoming)':
                index = -2
            else:
                index = -1
            if rel not in range_info:
                type = rel.split('.')[index]
            else:
                type = range_info[rel][index]
            
            rel_desc = schema_desc_template(type, rel, direction=direction.strip('(').strip(')')+ ' relation')
            explanation += f"{ix+1}. {rel_desc} "
    return None, explanation


def argmin(variable: str, attribute: str, sparql_executor):
    """
    Get the argmin of a variable
    :param variable: a variable
    :param relation: a relation
    :return: the argmin of a variable
    """
    if attribute not in variable_attributes_cache[variable]:
        raise ValueError("argmin: attribute must be an attribute of the variable")

    rtn_str = f": variable ##, which are instances of {variable.type}\n"
    new_variable = Variable(variable.type, f"(ARGMIN {variable.program} {attribute})")
    return new_variable, rtn_str

def lt(attribute: str, value, sparql_executor):
    rtn_str = f": variable ##, which are instances of {range_info[attribute][0]}"
    new_variable = Variable(range_info[attribute][0], 
                            f"(lt {attribute} {value})")
    return new_variable, rtn_str

def le(attribute:str, value, sparql_executor):

    rtn_str = f": variable ##, which are instances of {range_info[attribute][0]}"
    new_variable = Variable(range_info[attribute][0], 
                            f"(le {attribute} {value})")
    return new_variable, rtn_str

def ge(attribute:str, value, sparql_executor):
    rtn_str = f": variable ##, which are instances of {range_info[attribute][0]}"
    new_variable = Variable(range_info[attribute][0], 
                            f"(ge {attribute} {value})")
    return new_variable, rtn_str

def gt(attribute:str, value, sparql_executor):
    rtn_str = f": variable ##, which are instances of {range_info[attribute][0]}"
    new_variable = Variable(range_info[attribute][0], 
                            f"(gt {attribute} {value})")
    return new_variable, rtn_str


if __name__ == '__main__':
    from kg_querier.kg_querier import KGQuerier
    sparql_executor = KGQuerier(load_retriever=False)

    program = '(JOIN cvg.cvg_publisher.games_published_inv m.01n073)'
    processed_code = postprocess_raw_code(program)
    sparql_query = lisp_to_sparql(processed_code)
    clauses = sparql_query.split("\n")
    new_clauses = [clauses[0], "SELECT DISTINCT ?rel\nWHERE {\n?x ?rel ?obj .\n{"]
    new_clauses.extend(clauses[1:])
    new_clauses.append("}\n}")
    new_query = '\n'.join(new_clauses)
    out_relations = sparql_executor.execute_query(new_query)
    print(out_relations['rel'])