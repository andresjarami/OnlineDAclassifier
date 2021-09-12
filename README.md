# Experiments of the paper "Towards Discriminant Analysis Classifiers using Online Active Learning via Myoelectric Interfaces"

This repository describes the three experiments performed in the paper *Towards Discriminant Analysis Classifiers using 
Online Active Learning via Myoelectric Interfaces*. We implemented the experiments using 
[python 3.7](https://www.python.org/downloads/release/python-377/).


## Required libraries 
Numpy [https://numpy.org/install/](https://numpy.org/install/)

Pandas [https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html/](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html/)

Matplotlib [https://matplotlib.org/3.1.1/users/installing.html](https://matplotlib.org/3.1.1/users/installing.html)

Scipy [https://www.scipy.org/install.html](https://www.scipy.org/install.html)

Time [https://docs.python.org/3/library/time.html](https://docs.python.org/3/library/time.html)

Math [https://docs.python.org/3/library/math.html](https://docs.python.org/3/library/math.html)

Sklearn [https://scikit-learn.org/stable/install.html](https://scikit-learn.org/stable/install.html)

Itertools [https://pypi.org/project/more-itertools/](https://pypi.org/project/more-itertools/)

Random [https://pypi.org/project/random2/](https://pypi.org/project/random2/)

## Databases

1. [NinaPro5](http://ninaweb.hevs.ch/). This database should import to the folder [ninaDB5](Datasets/ninaDB5).
2. [Capgmyo_dbb](http://zju-capg.org/research_en_electro_capgmyo.html#download). This database should import to the folder [capgmyo_dbb](Datasets/capgmyo_dbb).
3. [MyoArmband](https://github.com/UlysseCoteAllard/MyoArmbandDataset). This database should import to the folder [MyoArmbandDataset-master](Datasets/MyoArmbandDataset-master).
4. [Long-Term 3DC](https://ieee-dataport.org/open-access/long-term-3dc-dataset). This database should import to the folder [longterm_dataset_3DC](Datasets/longterm_dataset_3DC).
5. [EMG-EPN-120](https://ieeexplore.ieee.org/abstract/document/8903136/?casa_token=RYo5viuh6S8AAAAA:lizIpEqM4rK5eeo1Wxm-aPuDB20da2PngeRRnrC7agqSK1j26mqmtq5YJFLive7uW083m9tT). 
This database should import to the folder [rawData](Datasets/EMG_EPN120_Dataset/allUsers_data/rawData). In addition, the file [main](Datasets/EMG_EPN120_Dataset/Detect_muscle_activity/main.m) 
should be executed to only segment the gesture data using the detect-muscle-activity's technique proposed by [Benalcazar et al. (2017)](https://ieeexplore.ieee.org/document/8247458). 

## Experiments

To reproduce our experiments, please perform the following steps:

1. We extracted the data from the five databases using the [DataExtraction](ExtractedData/DataExtraction.py) python file. This file was run over a personal computer (Intel® Core™ i5-8250U processor and 8GB of RAM).

2. The next step was performed over a supercomputer in [NCI Gadi](http://nci.org.au/our-services/supercomputing). The characteristics of Gadi are Intel Xeon Platinum 8274 (Cascade Lake), two physical processors per node, 3.2 GHz clock speed, and 48 cores per node. Documentation for NCI Gadi can be found [here](https://opus.nci.org.au/display/Help/Gadi+User+Guide).
For the experiment 1 and 2, we run over Gadi five batch files (one per database), which are in the folder [server_files](Experiments/server_files).


## Visualization of the two experiments:
After the experiments' execution, we use Jupyter notebooks to analyze and develop the graphs presented in the paper.

[Experiments](Experiments_visualization.ipynb) (jupyter notebook) or [Experiments](Experiments_in_Markdown/Experiments_visualization.md) (markdown file)
