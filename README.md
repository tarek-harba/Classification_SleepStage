# Machine Learning for Sleep Stage Classification using EEG measurements
In this project, I follow the recent [research paper](https://ieeexplore.ieee.org/document/9417097)
which introduces a novel neural network architecture that utilizes [EEG recordings](https://en.wikipedia.org/wiki/Electroencephalography)
in order to classify sleep stage (wake, N1, N2, N3, REM). This is relevant in the diagnosis of many illnesses that range
from neurological to cardiovascular. 
This project also makes for a great demonstration of time-series analysis.

**I implement the network from scratch** with minor adjustments
to reduce training time and accommodate missing details from the paper. Lastly, the implementation
is done in a modular manner, facilitating analysis and ablation study conduction. 

The network was trained on a cloud-gpu that is provided by [Runpod](https://www.runpod.io/) which was
preferred over the big-ones like AWS, GCP and Azure for its lower pricing.

## Sample of Data
Before discussing the network, it is worthwhile to look at what we will be dealing with. A sample of the
EEG measurements is shown below where each plot represents a 30-second that is taken from one of the 
datasets and falls under a certain class (of those mentioned earlier).
![samples.png](Readme%20Images%2Fsamples.png)
We can see the complexity of the problem as such signals comprise wide range of frequencies
which in turn, may require different feature extraction models for each frequency range. 
This is the key point in the research paper.

## Network Design
### Overall framework
The network is constructed using 3 modules which are:
* Multi-resolution recurrent neural network (MRCNN): Extracts different features from 
different frequency bands.
* Automatic Feature Recalibration (AFR): Models relations between features which improves learning.
* Temporal Context Encoder (TCE): Captures temporal dependencies in features and is inspired
by the Transformer model.

Moreover, the network utilizes a Class-Aware loss function (weighted cross-entropy) which takes
into account class imbalances. This improves network performance given a dataset with 
under-represented classes.

The overall model design is depicted below.
![overall framework.jpeg](Readme%20Images%2Foverall%20framework.jpeg)
The first segment to the left involves MRCNN and AFR whereas the middle section involves TCE.
The last segment uses a fully connected layer  that trims the input to get an output
that allows the application of softmax function, and in turn, give likelihood predictions for each class.

### Framework in Detail
More specifically, the **MRCNN** and **AFR** are shown below in more detail with the module to the left
representing MRCNN and the module to the right being the AFR.
![mrcnn+afr.jpeg](Readme%20Images%2Fmrcnn%2Bafr.jpeg)

As for the **TCE** that is shown below, the proposed design was used as a guide
that is not to be followed strictly. Those that are interested in my implementation can contact me and
I would be more than happy to share the details.
![TCE.jpeg](Readme%20Images%2FTCE.jpeg)


## Performance
In the paper, the authors trained the model in 20 folds for 100 epochs. Due to the cost
involved with renting a cloud GPU, my network was only trained for 40 epochs. Additionally, the authors
had 3 different datasets. Due to the redundancy involved with training and testing different datasets,
the tables below only show the results for [edf20](https://github.com/emadeldeen24/AttnSleep/blob/main/prepare_datasets/download_edf20.sh) dataset which is the smallest.

The performance results are presented exactly as the authors did, all of which can be seen below.
 
## Per-Class F1 Score
| **Per-Class F1 Score** |   W   |    N1 |    N2 |    N3 |   REM |
|------------------------|:-----:|------:|------:|------:|------:|
| Paper Implementation   | 89.7  |  42.6 |  88.8 |  90.2 |  79.0 |
|  My Implementation     | 89.76 | 59.25 | 87.61 | 89.82 | 84.91 |

## Overall Metrics
| **Overall Metrics**  | Accuracy |   MF1 |     κ |   MGm |
|----------------------|:--------:|------:|------:|------:|
| Paper Implementation |   84.4   |  78.1 |  79.0 |  85.5 |
| My Implementation    |   75.7   | 82.27 | 74.23 | 82.89 |


## Confusion Matrix - Per-Class Metrics
| **Paper Implementation** |  PR  |   RE |   F1 |   GM |
|--------------------------|:----:|-----:|-----:|-----:|
| W                        | 89.6 | 89.7 | 89.7 | 92.5 |
| N1                       | 47.1 | 39.1 | 42.8 | 61.6 |
| N2                       | 89.1 | 88.6 | 88.8 | 90.3 |
| N3                       | 90.7 | 89.8 | 90.2 | 94.1 |
| REM                      | 76.1 | 82.2 | 79.0 | 88.0 |

| **My Implementation** |  PR   |    RE |    F1 |           GM |
|-----------------------|:-----:|------:|------:|-------------:|
| W                     | 97.39 | 83.87 | 89.76 |        90.15 |
| N1                    | 88.22 | 46.78 | 59.25 |        64.65 |
| N2                    | 92.29 | 84.18 | 87.61 |        88.19 |
| N3                    | 98.57 | 83.44 | 89.82 |        90.47 |
| REM                   | 93.08 | 80.23 | 84.91 |        86.02 |

## Ablation Study
The results of the ablation from both the paper and self implementation are presented below.

Paper's Ablation Study             |  Self Implementation's Ablation Study
:-------------------------:|:-------------------------:
![paper ablation.jpeg](Readme%20Images%2Fpaper%20ablation.jpeg)  |  ![self ablation.png](Readme%20Images%2Fself%20ablation.png)

A sanity check that is fulfilled is the gradual improvement in performance from the orange
to cyan then to purple model. This means that the additional complexity is paying for itself.

One thing that is strange at first sight is the relatively good performance of the lighter grey and 
blue models. This is the case **because** of my choice of training for only 40 epochs which
meant that the simpler networks were able to learn enough whereas the more complicated networks
simply couldn't. My assessment is that if I were to train the models for 100 epochs, the general trend
of this network's ablation would mimic that of the paper's.

## Files Description
Below is the directory tree of this project with
a brief description of the potentially unclear files:
```
Classification_SleepStage/
├── Data_Handling/
│   ├── merge_data.py : merges downloaded data into a single file
│   ├── sleep_dataset.py : data object fed into network
│   └── visualize_data.py : used to generate plots of EEG readings
├── models/
│   ├── model_blue.py
│   ├── model_orange.py
│   ├── model_grey.py
│   ├── model_cyan.py
│   └── model_purple.py 
├── modules/ : modules used to construct final models
│   ├── afr.py
│   ├── conv.py
│   ├── mrcnn.py
│   └── tce.py 
├── performance/
│   ├── compare_models.py : used to generate ablation study figure
│   ├── metrics.py : called by performance_object.py to calculate performance metrics
│   ├── performance_object.py : an object that stores performance results
│   └── process_results.py : Displays results used in tables
├── preprocess/ : processing of downloaded data, as is from paper
│   ├── dhedfreader.py 
│   └── prepare_physionet.py
├── Readme Images/
├── Results/ : contains files representing performane of each model
├── main.py : used to train network on specifiec model in files
├── train.py : instructions for building folds and training process
└── calculate_weight.py : calculates weights used in the class-aware loss function

```

## Final Thoughts and Comments
The overall the performance of the self implemented network is surprisingly close to the author's given the
lighter training with only 40 epochs against 100.

Also, it may seem strange that in some metrics, my implementation scores better, but it is important to bear in mind
that more training may result in worse performance in some measures and better performance in others.