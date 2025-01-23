# CFDNNEval

The code repository for the paper *CFDNNEval : A comprehensive evaluation of neural network models  for computational fluid dynamics*

CFDNNEval is a comprehensive assessment of 12 operator learning-based NN models tailored to emulate the behavior of seven benchmark fluid dynamics problems. These benchmark problems encompass a diverse array of two-dimensional scenarios, ranging from Darcy flow, Taylor-Green vortex, and lid-driven cavity to tube flow, cylinder flow, two-phase flow, and the three-dimensional periodic hill flow. To facilitate rigorous testing, we establish 22 fluid dynamics datasets corresponding to these benchmark problems, 18 of which are newly generated using conventional numerical techniques such as the finite element method. Leveraging these datasets, we meticulously evaluate the performance of 12 advanced NN models, addressing challenges such as intricate geometry, nonlinearity, long-term prediction, multi-scale phenomena, multi-phase flow, and convection dominance. Our evaluation covers computational accuracy, efficiency, and fluid field visualization, providing a crucial reference for the adoption of NN models in fluid dynamics research.

## UnifiedPDESolvers

Original PyTorch implementation of UPS proposed in the paper "[UPS: Towards Building Foundation Models for PDE Solving via Cross-Modal Adaptation](https://arxiv.org/abs/2403.07187)". UPS is developed for solving diverse spatiotemporal PDEs defined over various domains, dimensions, and resolutions. It unifies different PDEs into a consistent representation space and processes diverse collections of PDE data using a unified network architecture that combines LLMs with domain-specific neural operators.

## Repository Structure

This repository includes the implementation of benchmark models with the following structure:

- **`./config`**: Contains model parameters organized into different YAML files based on the dataset and model.
- **`./generate_metadata_data.ipynb`**: Generate the metadata and PDE data of the task.
- **`./generate_text_embeddings.py`**: Generate text embeddings of the metadata.
- **`main.py`**: Contains the training and testing framework.
- **`task_configs.py`**: Set task configs.
- **`data_loaders.py`**: Dataset and dataloaders.
- **`embedder.py`**: Network.
- **`utils.py`**: Includes utility functions for data reading and network construction.

## Requirements
```
pip install -r requirements.txt
```
Note that the `attrdict` package might not be compatible for python 3.10 or newer versions. If getting `ImportError: cannot import name 'Mapping' from 'collections'`, change 
```
from collections import Mapping
```
to 
```
from collections.abc import Mapping
```

## Training models
1. Download [CFDNNEval Datasets](https://drive.google.com/drive/folders/1Ao9vfWjy1VTwTa-9N1axZlXlz1s6CjPW?usp=sharing) to `./datasets`
2. Generate the PDE metadata by 
```
generate_metadata_data.ipynb
```
3. Generate the text embeddings
```
python generate_text_embeddings.py
```
4. Use an existing configuration file or add a new one to `./configs`
5. Run training
```
python main.py --config configs/config_file_name.yaml 
```
Model checkpoints will be released later.



## License

MIT licensed, except where otherwise stated. See `LICENSE` file.
