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

# Whistle Annotation
The Utils folder contains a file called annotateWhistle.py, which can be used to quickly and easily annotate whistle data for training. All you need to do is create at least one folder in the WhistleDetector folder where the WAV files to be annotated are stored. This can be the AnnotationData folder, for example. This folder already contains a very short audio file with just one whistle for demonstration purposes.

To annotate a file, only the last line of annotateWhistle.py needs to be adapted. The parameter "dataset" stands for the name of the WAV file to be annotated (without extension) and the parameter "folder" for the folder in which the WAV file is located.
For example:
```
label_dataset(dataset = "Demo", folder = "AnnotationData")
```
If the annotateWhistle.py file is then executed, a matplotlib window opens after a while. In this window, the recognized whistles are marked by a white frame on the spectrogram. If you want to listen to a whistle again, you can do this by double left-clicking on it. To mark a recognized whistle as a real whistle, right-click on the position of the whistle. The frame changes from white to green. When all real whistles have been marked, the corresponding data can be exported by pressing the s key.

By pressing the l key, you can switch from annotation mode to edit mode. However, all previously annotated whistles will be lost. In edit mode, noise or whistling can be removed by right-clicking. The edited file can then be saved by pressing the c key.

### Key assignment:
|Key | Action|
|-------- | --------|
| l | Switch between edit and annotation mode (default)|
| c | save edited data file |
| s | export annotation data |
| double left click | play audio |
| right click | mark whistle as real [annotation mode (default)] / cut audio [edit mode]