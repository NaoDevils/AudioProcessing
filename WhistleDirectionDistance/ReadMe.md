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
