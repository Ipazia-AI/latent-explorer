
import re
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import networkx_temporal as tx
from numpy import ceil
from matplotlib.patches import FancyBboxPatch
    
def clearString(s):
    if not isinstance(s, str) or s == '':
        return s
    
    s = s.replace('_', ' ')
    s = s[0].upper() + s[1:]
    
    if len(re.findall('[A-Z][^A-Z]*', s)) > 0:
        s = ' '.join([word.capitalize().strip() for word in re.findall('[A-Z][^A-Z]*', s)])

    return s
    
def draw_temporal_graph(TG, suptitle = None):
    
    # Get the nodes for the inference layer
    inferece_nodes = TG[-1].nodes()
    
    # Compute the number of rows and columns
    ncols = 5 if len(TG) > 20 else 4
    nrows = ceil(len(TG) / ncols).astype(int)
    spare_suplots = (ncols * nrows) - len(TG)
    
    # Create the figure
    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize=(30, 5 * nrows)) # , constrained_layout=True  

    i, j = 0, 0
    for t, G in enumerate(TG):
        
        # Get the axis
        ax_ = ax[t] if nrows == 1 else ax[i, j]
        ax_.margins(0.1)  
        last = t + 1 == len(TG)

        # Draw the graph
        if G.number_of_nodes() < 8:
            pos = nx.shell_layout(G, scale = 1)
        else:
            pos = nx.circular_layout(G, scale = 1)

        # Draw the graph
        colors = {node: 'firebrick' if node in inferece_nodes else 'indianred' for node in G.nodes()}
        node_labels = {i: '\n'.join(clearString(att['feature']).upper().split()) 
                       if len(att['feature'].split()) < 5 else ' '.join(word.upper() if idx % 2 else f'{word.upper()}\n' 
                                                                     for idx, word in enumerate(clearString(att['feature']).split())) 
                  for i, att in G.nodes(data = True)}
        nx.draw(G, 
                pos = pos, 
                ax = ax_, 
                labels = node_labels,
                node_size = 5000 if G.number_of_nodes() < 8 else 3500, 
                with_labels = True, 
                alpha = 0.9,
                font_size = 10 if G.number_of_nodes() < 8 else 9, 
                edgecolors = list(colors.values()), 
                font_color = 'white', 
                node_color = list(colors.values()),
                edge_color = 'black', 
                arrowsize = 20, 
                width = 2)

        # Draw the edge labels
        edge_lables = {edge: ' '.join(word if idx % 2 else f'{word}\n' 
                                      for idx, word in enumerate(clearString(label).split())).strip() 
                       if len(clearString(label).split()) > 2 else label 
                       for edge, label in nx.get_edge_attributes(G,'label').items()}
        
        nx.draw_networkx_edge_labels(G, pos, ax = ax_, edge_labels = edge_lables, 
                                     font_size = 12 if G.number_of_nodes() < 8 else 10, 
                                     alpha = 0.7 if last else 1, font_color = 'black', bbox = dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))  
        
        # Add a border around the subplot
        p = FancyBboxPatch((.02, .02), width=.96, height=.96, boxstyle="round,pad=0.02", 
                       fill=False, edgecolor ='black', ls = 'dashed' if last else 'solid', alpha = 0.7 if last else 1,
                       linewidth=1 if last else 2, transform=ax_.transAxes, clip_on=False, zorder = 10)
        ax_.add_patch(p)
        
        # Set the title
        ax_.set_title(label = G.name if 'inference' not in G.name.lower() else 'Inference',
                      fontsize = 16,  color = 'black' if last else 'firebrick', alpha = 0.7 if last else 1)

        # Update the indices
        j += 1
        if ncols and j % ncols == 0:
            j = 0
            i += 1

    # Remove the spare subplots
    for ax in ax.flatten()[len(TG):len(TG) + spare_suplots]:
        ax.remove()
    
    # Add the suptitle
    if suptitle:
        fig.suptitle(suptitle, fontsize=28, fontweight='bold', color = 'firebrick', y = 1)
    
    # Adjust the layout
    fig.tight_layout(h_pad=2)
    
    # Close the figure
    plt.close(fig)
        
    return fig


def generate_temporal_graphs(df):
    
    # Iterate over the inputs
    temporal_graphs = list()
    for _, df_row in df.iterrows():
        
        # Crate the edge list
        edge_df = pd.DataFrame(df_row['hs'].items(), columns = ['layer', 'triple'])
        
        # Add the inference triples
        edge_df.loc[len(edge_df.index)] = ['INFERENCE', df_row['inference']] 
        
        # Unpack the triples
        edge_df = edge_df.explode('triple').reset_index(drop = True)
        unpacked = edge_df['triple'].apply(lambda triple: pd.Series(triple, index = ['source', 'relation', 'target']))
        edge_df = pd.concat([edge_df, unpacked], axis = 1).drop(columns = ['triple'])
        edge_df = edge_df.dropna()
        
        if edge_df.empty:
            continue
        
        # Prepare for the graph
        edge_df['relation'] = edge_df['relation'].map(lambda item: {'label': item})
        
        # Create the graph
        TG = tx.TemporalGraph(directed=True, multigraph=False, t = len(edge_df['layer'].unique()))
        TG.name = df_row['input']
        
        # Add the edges for each layer
        for idk, layer in enumerate(edge_df['layer'].unique()):
            layer_triples = edge_df[edge_df['layer'] == layer].copy()

            # Generate the node ids
            unique_nodes = set(layer_triples['source'].values).union(set(layer_triples['target'].values))
            node_mapping = {nodeName:idk  for idk, nodeName in enumerate(unique_nodes)}
            layer_triples['source'] = layer_triples['source'].map(lambda nodeName: node_mapping[nodeName])
            layer_triples['target'] = layer_triples['target'].map(lambda nodeName: node_mapping[nodeName])

            # Add the nodes and edges
            TG[idk].add_edges_from(layer_triples[['source', 'target', 'relation']].values)
            
            # Add the names to the nodes
            for node_id, attribute in TG[idk].nodes(data = True):
                attribute['feature'] = [nodeName for nodeName, idk in node_mapping.items() if idk == node_id][0]
                
            # Assign the name to the graph 
            TG[idk].name = f"Layer {layer.strip('L')}"
            
        temporal_graphs.append(TG)
    return temporal_graphs