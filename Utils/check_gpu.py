import tensorflow as tf

#####################
### Check for GPU ###
#####################
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f'Using TensorFlow {tf.__version__}, GPUs available? : {len(gpus)}')
if not gpus or len(gpus) < 1:
    correct = input("You are not using any GPU is this correct? [Y/n]")
    if correct.lower() == "n":
        exit(-1)
