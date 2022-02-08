# Decoupled Smoothing in Probabilistic Soft Logic

Experiments for "Decoupled Smoothing in Probabilistic Soft Logic".

## Probabilistic Soft Logic

Probabilistic Soft Logic (PSL) is a machine learning framework for developing probabilistic models. You can find more information about PSL available at the [PSL homepage](https://psl.linqs.org/) and [examples of PSL](https://github.com/linqs/psl-examples). 

## Documentation

This repository contains code to run PSL rules for one-hop method, two-hop method, and decoupled smoothing method for predicting genders in a social network. 
We provide links to the datasets (Facebook100) in the data sub-folder.

### Obtaining the data

This repository set-up assumes that the FB100 (raw `.mat` files) have been acquired and are saved the data folder. Follow these steps:
1. The Facebook100 (FB100) dataset is publicly available from the Internet Archive at https://archive.org/details/oxford-2005-facebook-matrix and other public repositories. Download the datasets.
2. Save raw datasets in placeholder folder data. They should be in the following form: `Amherst41.mat`.

### Set permissions

Make sure that permissions are set so you can run the run scripts: 
```
chmod -R +x *
```

### Reproducing results

#### Step 1: Generate input files

To reproduce the results, first need to generate the predicate txts, run `./generate_data.sh {school name}`. It will automatically generate the files required to run the PSL models as well as the files to run the baseline model. 

For example, to generate data using Amherst college as dataset, simply run `./generate_data.sh Amherst41`.



**Step 2: Run PSL models**

**Simple Exeucution** 

To reproduce the results of a specific PSL model, run  `./run_all.sh {data} {method dir}`. This will run a selected method for all random seeds at all percentages. 

This takes the following positional parameters: 
* data: what datafile you would like to use
* method dir: this is the path to the directory you'd like the run


For example, to reproduce the result for method `one-hop` using the Amherst college as dataset, simply run `./run_all.sh Amherst41 cli_one_hop`.

**Advanced Execution**

If you need to get results for a more specific setting, run `./run_method.sh {data} {random seed} {precent labeled} {eval|learn} {method dir}`. It runs a selected method for a specified seed for a specified percentage for either learning or evaluation.

This takes the following positional parameters: 
* data: what datafile you would like to use
* random seed: what seed to use
* percent labeled: what percentage of labeled data
* {learn|eval}: specify if you're learning or evaluating
* method dir: this is the path to the directory you'd like the run


The output will be written in the following directory:
`../results/decoupled-smoothing/{eval|learn}/{method run}/{data used}/{random seed}/`

The directory will contain a set of folders for the inferences found at each percent labeled, named `inferred-predicates{pct labeled}`.
The folder will also contain the a copy of the `base.data`, `gender.psl`, files and output logs from the runs.

#### Step 3: Run baseline Decoupled Smoothing model

To run the baseline decoupled smoothing model, run `baseline_ds.py`. It will generate a csv file contains the results of the baseline model named `baseline_result.csv`.


### Evaluation
To run the evaluation of each models, run `evaluation.py`, which will generate the two plots in Figure 3 in the paper.


### Requirements
These experiments expect that you are running on a POSIX (Linux/Mac) system. The specific application dependencies are as follows:

* Python3
* Bash >= 4.0
* PostgreSQL >= 9.5
* Java >= 7


### Citation

All of these experiments are discussed in the following paper:

```
@inproceedings{chen:mlg20,
    title = {Decoupled Smoothing in Probabilistic Soft Logic},
    author = {Yatong Chen and Byran Tor and Eriq Augustine and Lise Getoor},
    booktitle = {International Workshop on Mining and Learning with Graphs (MLG)},
    year = {2020},
    publisher = {MLG},
    address = {Virtual},
}
```


