import re
import numpy as np

def _negativeOperator(expressions, verbose = False):
    facts = []
    for negative in expressions:
        negative = negative[:-1] # remove the last parenthesis
        for fact in re.split('∧|∨', negative):
            aux_matches = re.search('is|has', fact, re.IGNORECASE)
            if aux_matches:
                fact = fact[:aux_matches.end()] + 'Not' + fact[aux_matches.end():]
            else:
                fact = 'Not' + fact
            fact = fact.strip(' ¬(')
            facts.append(fact)
            
        if verbose:
            print('NEGATIVE:', facts, " --> PARSED:", facts)
    return facts


def _singleObjectFact(relation, object):
    relationComponents = re.findall('[A-Z][^A-Z]*', relation)

    if object.lower() in ['true', 'false']:
        object2 = object 
        
        if len(relationComponents) > 2:
            obj_pos =  1 if relationComponents[1].lower() not in ['of', 'in'] else relationComponents.index('Of' if relationComponents[1].lower() == 'of' else 'In') + 1
            rel = ' '.join(relationComponents[:obj_pos])
            object1 = ' '.join(relationComponents[obj_pos:])
        else:
            rel = 'is'
            object1 = relation 
    elif len(relationComponents) > 1:
        obj_pos = -1  if relationComponents[0].lower() not in ['is', 'has'] else relationComponents.index('Is' if relationComponents[0].lower() == 'is' else 'Has') + 1
        if 'not' in relationComponents[obj_pos].lower():
            obj_pos += 1

        object1 = object
        rel =  ' '.join(relationComponents[:obj_pos]).replace('_', ' ')
        object2 =  ' '.join(relationComponents[obj_pos:]).replace('_', ' ')
    else:
        object1 = object
        rel = 'is'
        object2 = relationComponents[0]

    return rel, [object1, object2]

def generateSPOTriples(items, verbose = False):

    # isCity(New York City) --> (New York City, is, City) 
    
    # Iterate over the propositions
    triples = []
    for item in items:
        if not isinstance(item, str):
            continue
        
        if verbose:
            print(f"\nRAW:", item)
        
        # Handle the global negation operator ¬() if present 
        global_negation = re.findall(r'¬\(.*\)', item)  # .replace('Not', '¬')
        if len(global_negation) > 0:
            expanded_negative_facts = _negativeOperator(global_negation)
            
            # Remove the negative facts from the original string
            for neg in global_negation:
                item =  item.replace(neg, '')

        # Split the text with the delimiters
        facts = re.split('∧|∨|≠', item)
        
        # Append the unpacked negative facts if present
        if len(global_negation) > 0:
            facts.extend(expanded_negative_facts)
        
        # Parse the facts
        for fact in facts:
            
            # Skip empty facts
            fact = fact.strip()
            if fact == '':
                continue
            
            if verbose:
                print('--> LOGIC:', fact)

            # Extract the parts of the fact
            components = re.split(r'\(', fact)
            
            # Relation
            rel = components[0].strip()

            if len(components) > 2:
                sub_objects = ','.join(components[1:])
                objects = re.split(r',', sub_objects)
            elif len(components) == 2:
                objects = re.split(r',', components[1])
            elif len(components) == 1:
                continue
            else:
                raise NotImplementedError(f'Number of components ({len(components)}) not handled')
            
            objects = [obj.strip(' )') for obj in objects]

            # Clean the strings
            objects = [obj.strip() for obj in objects]
            
            if '¬' in rel:
                rel = rel.replace('¬', '')
                pos = re.search('is|has', rel, re.IGNORECASE)
                if pos:
                    rel = rel[:pos.end()] + 'Not' + rel[pos.end():]
                else:
                    rel = 'Not' + rel
                            
            # Special case with true/false object
            tf_matching = np.argwhere([obj.lower() in ['true', 'false'] for obj in objects]).flatten()
            binary_flag = False
            if len(objects) > 1 and len(tf_matching) > 0:
                obj = objects[tf_matching[0]]
                binary_flag = True
                
                # Modify the relation
                if obj.lower() == 'false':
                    pos = re.search('is|has', rel, re.IGNORECASE)
                    if pos:
                        rel = rel[:pos.end()] + 'Not' + rel[pos.end():]
                    else:
                        rel = 'Not' + rel
                objects.remove(obj)
            
                if len(objects) == 1:
                    rel, objects = _singleObjectFact(rel, objects[0])
                else:
                    rel = rel + objects[0]
                    objects = objects[1:]
                    #raise Exception(f'Unhandled case with {len(objects)} objects ({objects}) --> raw: {fact}')
            
            # Special case
            if len(objects) > 1 and '∧' in objects[0]: 
                objects = objects[0].split('∧') + objects[1:]
            elif len(objects) == 1:
                rel, objects = _singleObjectFact(rel, objects[0])
            
            # Structure the triple for each object
            for obj in objects[1:]:
                
                # Skip the etc
                if 'etc' in obj.lower():
                    continue
  
                # Object order
                if  not binary_flag and ('is' in rel.lower() or 'has' in rel.lower()):
                    triple = (obj, rel, objects[0])
                else:
                    triple = (objects[0], rel, obj)
 
                # Save the SPO triple
                triples.append(triple)
        
                if verbose:
                    print('\tSPO:', triple[0], '<-->', triple[1], '<-->', triple[2])
    return triples