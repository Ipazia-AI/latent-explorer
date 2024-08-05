
import pandas as pd
from os import path, makedirs
from typing import List, Dict
from networkx_temporal import TemporalGraph

# LOCAL IMPORTS
from .utils.triples import generateSPOTriples
from .utils.graph import generate_temporal_graphs, draw_temporal_graph
    
class TempoGrapher:
    
    def __init__(self, results: List[Dict[str, str]]):
        self.results = results
        self.graphs = self._generate_graphs()

    def _generate_graphs(self) -> List[TemporalGraph]:
        
        # Extract the facts from the results
        df = pd.DataFrame([{
            'input': output['STATS']['INPUT'], 
            'inference': output['_INFERENCE'],
            'hs': output['HS']} 
                            for output in self.results])

        # Generate the SPO triples
        extract_facts = lambda output: output['facts'] if isinstance(output, dict) and 'facts' in output.keys() else list()
        df['inference'] = df['inference'].apply(lambda text: generateSPOTriples(extract_facts(text))) 
        df['hs'] = df['hs'].apply(lambda layer_output: {layer: generateSPOTriples(extract_facts(text)) for layer, text in layer_output.items()}) 
        
        # Perform the knowledge graph generation
        temporal_graphs = generate_temporal_graphs(df = df[['input', 'inference', 'hs']])

        return temporal_graphs
    
    def get_graphs(self) -> List[TemporalGraph]:
        return self.graphs
    
    def save_graphs(self, folder_path:str):
        
        if len(self.graphs) == 0:
            print("No graphs to save!")
            return
        
        # Create the output folder
        folder_path = path.join(folder_path, 'graphs')
        if not path.exists(folder_path):
            makedirs(folder_path)
    
        # Iterate over the claims
        for graph in self.graphs:

            # Create the figure
            fig = draw_temporal_graph(graph, suptitle = graph.name)

            # Create the file name
            cleared_name = graph.name.replace(' ', '_').replace('/', '_').replace('"', '')
            cleared_name = cleared_name[:30] if len(cleared_name) > 30 else cleared_name
            
            # Save the figure
            fig.savefig(path.join(folder_path, f'{cleared_name}.pdf'))
        
        print(f"Graphs saved in {folder_path}")
    