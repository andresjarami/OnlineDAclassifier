# Experiments of the paper "Towards Discriminant Analysis Classifiers using Online Active Learning via Myoelectric Interfaces"

This repository describes the three experiments performed in the paper *Towards Discriminant Analysis Classifiers using 
Online Active Learning via Myoelectric Interfaces*. We implemented the experiments using 
[python 3.7](https://www.python.org/downloads/release/python-377/).

*NOTE:* This [link](https://anonymous.4open.science/r/Discriminant_Classifiers_Online_Active_learning-5191/README.md) is an anonymous git version of this repository.



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

1. Import [NinaPro5](http://ninaweb.hevs.ch/) to the folder [ninaDB5](Datasets/ninaDB5).
2. Import [Capgmyo_dbb](http://zju-capg.org/research_en_electro_capgmyo.html#download) to the folder [capgmyo_dbb](Datasets/capgmyo_dbb).
3. Import [MyoArmband](https://github.com/UlysseCoteAllard/MyoArmbandDataset) to the folder [MyoArmbandDataset-master](Datasets/MyoArmbandDataset-master).
4. Import [Long-Term 3DC](https://ieee-dataport.org/open-access/long-term-3dc-dataset) to the folder [longterm_dataset_3DC](Datasets/longterm_dataset_3DC).
5. Import [EMG-EPN-120](https://ieeexplore.ieee.org/abstract/document/8903136/?casa_token=RYo5viuh6S8AAAAA:lizIpEqM4rK5eeo1Wxm-aPuDB20da2PngeRRnrC7agqSK1j26mqmtq5YJFLive7uW083m9tT). 
to the folder [rawData](Datasets/EMG_EPN120_Dataset/allUsers_data/rawData). In addition, the file [main](Datasets/EMG_EPN120_Dataset/Detect_muscle_activity/main.m) 
should be executed to segment only the gesture data using the detect-muscle-activity's technique proposed by [Benalcazar et al. (2017)](https://ieeexplore.ieee.org/document/8247458). 

## Experiments

To reproduce our experiments, please perform the following steps:

1. To extract the feature sets from the five databases using the [DataExtraction](ExtractedData/DataExtraction.py) python file. This file was run over a personal computer (Intel® Core™ i5-8250U processor and 8GB of RAM).

2. For the experiment 1 and 2, to run over Gadi five batch files (one per database), which are in the folder [server_files](Experiments/server_files).

Note: The step 2 was performed over a supercomputer in [NCI Gadi](http://nci.org.au/our-services/supercomputing). The characteristics of Gadi are Intel Xeon Platinum 8274 (Cascade Lake), two physical processors per node, 3.2 GHz clock speed, and 48 cores per node. Documentation for NCI Gadi can be found [here](https://opus.nci.org.au/display/Help/Gadi+User+Guide).


## Visualization of the two experiments:
After the experiments' execution, we use Jupyter notebooks to analyze and develop the graphs presented in the paper.

[Experiments](Experiments_visualization.ipynb) (jupyter notebook) or [Experiments](Experiments_in_Markdown/Experiments_visualization.md) (markdown file)
