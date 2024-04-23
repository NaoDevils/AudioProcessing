# Whistle Config Parameters:

# General Config:
prefix = "x_log10_abs"
pycharm = False
batch_size = 2048

# Dataset Generation Config
n_clusters = 32
n_neighbours = 256
use_oversampling = True
use_undersampling = True
read_data_from_txt = False

# Training Config
max_epochs = 45
load_model = False
model_name = "WhistleNetMk9.h5"
desired_precision = 0.99
clear_checkpoints = True
show_model_summary = False