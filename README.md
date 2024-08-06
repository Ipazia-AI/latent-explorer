![Logo](https://github.com/Ipazia-AI/latent-explorer/raw/main/images/logo.png)

[![Static Badge](https://img.shields.io/badge/Licence-blue?label=CC%20BY-NC-SA%204.0)](http://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Dynamic TOML Badge](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FsaturnMars%2Flatent-explorer%2Fmain%2Fpyproject.toml&query=%24.project.version&label=release&color=green)](https://pypi.org/)
[![Static Badge](https://img.shields.io/badge/website-online-green)](https://github.com/Ipazia-AI)
[![Static Badge](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2404.03623-orange)](
https://doi.org/10.48550/arXiv.2404.03623)


Latent-Explorer is the Python implementation of the framework proposed in the paper [*Unveiling LLMs: The Evolution of Latent Representations in a Dynamic Knowledge Graph*](https://arxiv.org/abs/2404.03623) to appear in the 1<sup>st</sup> [Conference of Language Modeling](https://colmweb.org/index.html) (COLM).

## Overview
This framework analyses the process of factual knowledge resolution of Large Language Models (LLMs), representing its dynamics through graphs.
Using activation patching, it decodes the semantics, in the form of factual information, from the LLM latent representations (also known as residual stream or vector space) during the model's inference for the task of claim verification on entire input sentences. 
This framework can be used to study the LLMs' latent representations for several aspects, such as (i) which factual knowledge LLMs use to assess the truthfulness of factual claims, (ii) how this factual knowledge evolves across hidden layers, and (iii) whether there are any distinctive patterns in this evolution.

![Framework](https://github.com/Ipazia-AI/latent-explorer/raw/main/images/framework.png)

![Contribution](https://github.com/Ipazia-AI/latent-explorer/raw/main/images/contribution.png)


## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the Python package

```bash
pip install latent-explorer
```

or download [repository](https://github.com/Ipazia-AI/latent-explorer) and install the package with 
` pip install -e . `

## Demo
The folder `tutorial` includes a script showcasing the pipeline [`tutorial/script.py`](./tutorial/script.py)

*Since this framework performs multiple language model inferences, using a GPU is recommended ([see CUDA](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html))

## Usage

### Import the package
```python
import latent_explorer
```

### (1.a) Initialize the application with the LLM and the inputs
```python
explorer = latent_explorer.LatentExplorer(
  model_name = "meta-llama/llama-2-7b-chat-hf", 
  inputs = ["The capital of France is Paris"]
)
```
### (1.b) Prepare the textual prompts
```python
explorer.generate_prompts(verbose = True)
```

### (2) Perform the inference and get the hidden states
```python
explorer.inference(parse_output = True, output_hidden_states = True)
```

### (3.a) Probe each hidden states
```python
results = explorer.probe_hidden_states()
```

### (3.b) Save the textual results
```python
latent_explorer.save_results(results, folder_path = "outputs")
```

### (4.a) Generate the dynamic knowledge graphs
```python
tg = latent_explorer.TempoGrapher(results)
```

### (4.b) Get the graphs
```python
graphs = tg.get_graphs()
```

### (4.c) Generate and save the graphical figures
```python
tg.save_graphs(folder_path = "outputs")
```

## Language models available
This package inherits all of the LLMs supported by the [LitGPT](https://github.com/Lightning-AI/litgpt/blob/main/tutorials/download_model_weights.md) package.
This framework works with instruction-tuned language models, such as those named with the suffixes "inst", "instruction", or "chat". 

```python
models = latent_explorer.utils.all_supported_models()
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgements
This implementation is powered by [LitGPT](https://github.com/Lightning-AI/litgpt), conceptualised, designed and developed by [Marco Bronzini](https://www.linkedin.com/in/bronzinimarco).

This work has been funded by [Ipazia S.p.A.](https://ipazia.com)

## Citation
If you use this package or its code in your research, please cite the following work:

```bibtex
@misc{bronzini2024unveiling,
  title         = {Unveiling LLMs: The Evolution of Latent Representations in a Dynamic Knowledge Graph}, 
  author        = {Marco Bronzini and Carlo Nicolini and Bruno Lepri and Jacopo Staiano and Andrea Passerini},
  year          = {2024},
  eprint        = {2404.03623},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2404.03623}
}
```
## License
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
