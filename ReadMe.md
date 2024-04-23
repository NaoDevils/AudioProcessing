This repository contains all the necessary scripts and functions to create neural networks for audio processing like the ones we use.
Help functions / scripts can be found in the Utils folder. There is also a script to convert the created model files into tflite files.
All other scripts or Jupyter notebooks can be found sorted by topic in their respective folders.

# Requirements
There are several ways to install the dependencies of the Python scripts provided here. 
## Conda
### create environment with conda
conda env create -p ./venv -f conda_env.yml

## Requirements.txt
### create environment with virtualenv
pip install virtualenv

virtualenv venv

venv/Scripts/activate

pip install -r requirements.txt