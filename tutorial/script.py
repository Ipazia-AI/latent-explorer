from os import path

from latent_explorer import LatentExplorer, TempoGrapher
from latent_explorer.utils import save_results, all_supported_models

data = [
    "Empress Matilda moved to Germany as a child",
    "Robin was murdered by the Joker in a 1989 book",
    "Bojack Horseman's creator is also American"
]

if __name__ == '__main__':
    
    # List all supported models
    models = all_supported_models(verbose = True) 

    # APP INITALIZATION: Initialize the application
    explorer = LatentExplorer(model_name = "meta-llama/Llama-2-7b-chat-hf", inputs = data, hf_access_token = None) 

    # APP INITALIZATION: Generate the textual prompts (encompassing the system prompt, examples, and input text)
    explorer.generate_prompts(verbose = True)
    
    # (1) Basic inference (generated text, hidden states)
    explorer.inference(parse_output = True, output_hidden_states = True, verbose = False)

    # (2) Patched inferences (for each hidden states)
    results = explorer.probe_hidden_states(verbose = False)

    # (3) Save the results
    output_folder = path.join("outputs", *explorer.get_model_name())

    # (3a) Save the statistics
    explorer.save_stats(folder_path = output_folder)
     
    # (3b) Save the results as a JSON file
    save_results(results, folder_path = output_folder)
    
    # (4) Generate the temporal knowledge graphs
    tg = TempoGrapher(results)
    graphs = tg.get_graphs()

    # (5) Save the graphs
    tg.save_graphs(folder_path = output_folder)