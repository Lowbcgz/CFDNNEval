# CFDNNEval

The code repository for the paper *CFDNNEval : A comprehensive evaluation of neural network models  for computational fluid dynamics*

CFDNNEval is a comprehensive assessment of 12 operator learning-based NN models tailored to emulate the behavior of seven benchmark fluid dynamics problems. These benchmark problems encompass a diverse array of two-dimensional scenarios, ranging from Darcy flow, Taylor-Green vortex, and lid-driven cavity to tube flow, cylinder flow, two-phase flow, and the three-dimensional periodic hill flow. To facilitate rigorous testing, we establish 22 fluid dynamics datasets corresponding to these benchmark problems, 18 of which are newly generated using conventional numerical techniques such as the finite element method. Leveraging these datasets, we meticulously evaluate the performance of 12 advanced NN models, addressing challenges such as intricate geometry, nonlinearity, long-term prediction, multi-scale phenomena, multi-phase flow, and convection dominance. Our evaluation covers computational accuracy, efficiency, and fluid field visualization, providing a crucial reference for the adoption of NN models in fluid dynamics research.

## Datasets

We provide the benchmark datasets we used in the paper. The data generation configuration can be found in the paper.

* [CFDNNEval Datasets](https://drive.google.com/drive/folders/1Ao9vfWjy1VTwTa-9N1axZlXlz1s6CjPW?usp=sharing)

## Repository Structure

This repository includes the implementation of benchmark models with the following structure:

- `./config`: Contains model parameters organized into different YAML files based on the dataset and model.
- `./model`: Includes the network structure implementations for each model.
- `./scripts`: Contains all training and inference scripts for the benchmark models.
- `plot.ipynb`: Includes code for fluid field visualization.
- `train.py`: Contains the training framework.
- `utils.py`: Includes utility functions for data reading and network construction.

## Dependencies

Create a conda environment and install dependencies:

* Python 3.9
* CUDA 11.8
* PyTorch 2.1.2

and others dependencies.

```bash
# create environment
conda create -n CFDNNEval python=3.9
conda activate CFDNNEval 

# install pytorch
conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# others
pip install pyyaml h5py pandas matplotlib tensorboard
```

## Train and Test.
To run experiments, one should first download the datasets and correctly configure the dataset path in the corresponding configuration files. The configuration files are in YAML format and located in the `./config` folder, organized by `model_name/fluid_name`. 

The training command follows the rules outlined below: 

```bash
python train.py $CONFIG_FILE_PATH -c $CASE_NAME
```


and the test command follows by:

```bash
python test.py $CONFIG_FILE_PATH -c $CASE_NAME --save_result
```
The config file contains all the parameters needed for model training and testing, while `-c $CASE_NAME` should further select the interested physics case.

## License

MIT licensed, except where otherwise stated. See `LICENSE` file.