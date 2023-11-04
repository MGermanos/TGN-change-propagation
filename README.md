# To Change or Not to Change? Modeling Software System Interactions Using Temporal Graphs and Graph Neural Networks: a Focus on Change Propagation
## Overview

**Context:** The world is quickly adopting new technologies and evolving to rely on software systems for the simplest tasks. This prompts developers to expand their software systems by adding new product features. However, this expansion should be cautiously tackled to prevent the degradation of the quality of the software product. 

**Objective:** One challenge when modifying code - whether to patch a bug or add a feature- is knowing which components will be affected by the change and amending possible misbehavior. In this context, the study of change propagation or the impact of introducing a change is needed. By investigating how changing one component may impact the functionality of a dependency (another component), developers can prevent unexpected behavior and maintain the quality of their system. 

**Methods:** In this work, we tackle the change propagation problem by modeling a software system as a temporal graph where nodes represent system files and edges co-changeability, i.e., the tendency of two files to change together. The graph representation is temporal so that nodes and edges can change with time, reflecting the addition of files in the system and changes in dependencies. We then employ a Temporal Graph Network and a Long Short-Term Memory model to predict which other files will be impacted by a modification performed on a file.

**Results:** We test our model on software systems of different functionality, size, and nature. 
We compare our results to other published work, and our model shows a significantly higher ability to predict files impacted by a change. 

**Conclusion:** The proposed approach effectively predicts change propagation in software systems and can guide developers and software engineers in planning the change and estimating the cost in terms of time and money.

## Running the Experiments
### Requirements
Dependencies (with python >= 3.8.8):

```{bash}
pandas==1.4.3
numpy==1.19.5
tensorflow==2.14.1
```
### Dataset
This repository offers the datasets used in our work. Navigate to the following subfolders:
 `Shuffled Data\Iterator (0 to 4)\ChangeSets\Software name (i.e., alamofire)` To get to the change sets of the considered system for a given shuffle.
 Note that the shuffling occurs to the order of the files in a given commit. For example, the following change set $file A, file B, file C$ can be shuffledin this manner $file C,  file A,  file B$. But the chronological order of the change set is preserved.

 ### Executing the code
 The code of the model can be found at `TGN_model.py`. 
 To start, set which software systems you would like to test by specifying them at `line 32` in the variable `projects`.
 Then, set the parameters of the model starting from `line 70` till `line 75`.
 After making these adjustments, the code can be executed.

 ### Reading the results
The code will output into files the results of the computations. 
In the directory `Results\Metrics\directed_"project name"_results.csv`, you can find the performance metrics of every run. The columns of the file are divided as follows:
`type of representation graph, sensitivity, specificity, PPV, Gmean, F-measure, Accuracy, MCC, AUC`. Please note that the first colummn will always contain "directed" as that is the chosen representation for this work.
In the directory `Results\ConfMatrix\directed_"project name"_results"iteration number".csv`, you can find the confusion matrix of the predictions on every change set in the system. The columns of the files are divided as follows:
`True positives, True Negatives, False positives, False negatives`.
