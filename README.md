# Introduction
In this study we aim to analyse the effect that padding the amino acid sequences has on the performance of deep learning models. Specifically, we have chosen a hierarchical EC number prediction task to carry out the analysis. We use three different architectures (only_denses, 1_conv, stack_conv) to check if they affect the padding effect.

The data used in this study consists on all the reviewed protein sequences of the taxonomy Archaea from Uniprot version 2019_05.

The project is almost entirely coded in Python 3.6.7.  R version 3.4.4 has been used for building the explanatory linear models (notebooks 07 and 09). 

## Structure
- Functions are defined in the src/ folder (.py and .R files)
- The workflow of the analysis is applied through Jupyter Notebooks in the notebooks/ folder. The files are preceded by a number that indicates the chronological order of their execution.
- raw_data/ folder contains the CSV file with reviewed Uniprot entries for Archaea.
- data/ is the folder where intermediate and final results will be stored when running the notebooks.

## Workflow (notebooks)
### 1. Data preprocessing
- 00_creating_data.ipynb
- 01_pre_statistics.ipynb
### 2. Models training and results processing
- 02_task1_training.ipynb
- 03_task1_comparison.ipynb
- 04_task2_training.ipynb
- 05_task2_comparison.ipynb
### 3. Performance metrics and graphical representation
- 06_comparing_architectures.ipynb
- 07_linear_models_metrics.ipynb
- 08_activations_sequences_pca.ipynb
- 09_linear_models_pcs.ipynb

## System requirements
The runs have been executed on the following hardware from the B2SLab (Universitat Politecnica de Catalunya):
- *tob*:
      8 threads, 32GB RAM, NVIDIA TITAN Xp GeForce GTX 1070

- *lapsus*:
      12 threads, 32GB RAM, 2 x NVIDIA GeForce GTX 1070

## Considerations
- The absPath variable at the beginning of each notebook and src file should be changed once the repository is cloned to the correct path in each case.
- Notebooks 02-04 are implemented to analyse only one architecture each time. The block of variables/parameters in the second cell corresponding to the studied architecture should be uncommented (and the rest, commented) prior to running the notebook.
- In src/Target, all the different types of padding tested in this study are implemented.
- Execution without GPU of notebooks 02-05 and 08 may require considerable time and it is not recommended.
