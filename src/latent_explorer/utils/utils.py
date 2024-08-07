from json import dumps
from re import split
import dirtyjson
import numpy as np
import pandas as pd

def generate_prompt_template(sys_prompt = None, examples: list[dict] = [], input_text:str = '', role_sys_available = False) -> str:
    conversation = []
    
    # Add the system prompt
    if role_sys_available and sys_prompt is not None:
        conversation.append({"role": 'system', 'content': sys_prompt})
    else:
        conversation.append({"role": 'user', 'content': sys_prompt})
    
    # Add the in-context examples
    for pair in examples:
        claim, output = list(pair.items())[0]
        
        if not role_sys_available and len(conversation) == 1: # User input: Add the claim to the system instruction if the system role is not available
            conversation[0]['content'] += f"\n\n{claim}"
        else:  # User input: Add just the example claim to conversation if the system role is available
            conversation.append({"role": 'user', 'content': claim})
        
        # Assistant output
        conversation.append({"role": 'assistant', 'content': dumps(output, ensure_ascii = False)})
    
    # Add the input text
    if input_text:
        conversation.append({"role": 'user', 'content': input_text})

    return conversation

def text2json(output, max_attempts = 3, verbose = True):
    
    # Attempt to parse the output as JSON with some error handling
    num_attempts = 0
    while num_attempts < max_attempts:
        
        # Attempt to parse the output as JSON
        try:
            return dirtyjson.loads(output)
        
        # Handle the parsing error
        except dirtyjson.error.Error as e:
            
            # Return the output if the text is empty
            if len(e.doc) == 0:
                return output
            
            # Visualize the warning and error
            if verbose: 
                print('\t' * num_attempts + f'[WARNING] Parsing Error ({num_attempts + 1}° attempt):', e, '-->', 
                      f'[{e.doc[e.pos].upper()}]{e.doc[e.pos + 1:][:5]}...' if e.pos + 5 < len(e.doc) else f'[{e.doc[-1]}]')
                
                # Last attempt: Print the final output
                if num_attempts == (max_attempts - 1):
                    print('\t' * (num_attempts + 1), '-->', output, "\n")
            
            # FIX A: Error at the last character            
            if e.pos < len(e.doc): 
                if e.doc[e.pos] == '"' or e.doc[e.pos - 1] == '"':
                    output = output[:e.pos] + "]}"
            else:
                output += "]}" 
                
            # FIX B: a fully unstructured string
            if e.pos <= 1:
                splitted_output = [item.strip() for item in output.split('\n') if item.strip()]
                
                if len(splitted_output) > 1:
                    output = splitted_output[1] # skip the first component (e.g., assistant:\n ...)
                else:
                    return output.strip()
                    
            # Increment the attempt counter
            num_attempts += 1
        
        # Handle the general exception
        except Exception as e:
            print('[WARNING] Parsing General Error:', e)
            return output
    return output

def print_title(title, char = "-", length = 150):
    pad = length - len(title) - 2
    l_pad = pad // 2
    r_pad = (pad // 2) + (pad % 2)
    
    print("\n" + char*length)
    print(char * l_pad, title, char * r_pad)
    print(char*length + "\n")
    
def clean_token(token) -> str:        
    if len(token) <= 1:
        return token
    
    # Remove the special characters added by the tokenizer (artifacts of the tokenization process)
    special_chars = ['▁', 'Ġ']

    return token.strip(''.join(special_chars))
    
def search_subvector(vector, sub):
    
    # Clean the token
    vector = [clean_token(token) for token in vector]
    sub = [clean_token(token) for token in sub]
    
    # Reshape the vector and sub-vector
    vector = np.array(vector).reshape(1, -1)
    sub = np.array(sub).reshape(-1, 1)
    
    # Element-wise comparison
    comparison = np.argwhere(vector == sub)

    # Convert the comparison to a DataFrame and group by the first dimension
    df = pd.DataFrame(comparison)

    # Group by the first dimension and merge the second dimension
    df = df.groupby(by = 0)[1].apply(list)
    df = df.map(np.max).to_list()
    
    # Check if entire sub-vector is found in the vector, otherwise raise an error
    if len(df) == len(sub):
        return df
    else:
        print(f'\nORIGINAL: {sub[:, 0]}\nFOUND: {dict(zip(df, vector[0, df[0]:]))}')
        raise LookupError("The sub-vector is not found in the vector.")