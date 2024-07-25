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

Some whistle-containing audio files that can be annotated with this tool can be found [here](https://tu-dortmund.sciebo.de/s/hDiglhXxhO0JCB6).

# Whistle Direction and Distance Annotation
The Utils folder contains a file called annotateWhistleDirectionDistance.py, which can be used to quickly and easily annotate whistle distance and direction data for training, test and evaluation. You need only download the corresponding [dataset](https://tu-dortmund.sciebo.de/s/XXrULjGMD53JdqG) and copy the WAV files into the corresponding train, test, or evaluation folder. The demo folder already contains a very short audio file with just one whistle for demonstration purposes only.

To generate the annotated direction and distance data of the whistles, you need to edit the string in line 1006, which specifies one of the four folders (Train, Test, Eval, and Demo). By default, it contains the string "Demo". After this, you can simply run the script to generate all the files needed for training and testing our neural networks.
For example:
```
statistic = generate_dataset("Demo")
prettyPrintStatistic(statistic)
```
The generated statistics of the dataset can help balance your dataset and give a more detailed overview of it.

The annotateWhistleDirectionDistance.py file also contains a method to check the audio files for broken mics and a debug method that visualizes the audio files as spectrograms and marks detected whistles. This debug method mostly works like the whistle annotation tool and has the same shortcuts. 
For example:
```
check_for_broken_mics("Test")
debug_dataset("Hulk_Lab_Test_02", "Eval")
```
