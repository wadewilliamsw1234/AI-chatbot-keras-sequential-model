# Chatbot built using real chat data and pretrained model

This project is compiled with python for the purpose of  pre-trained USE model. The code is intended to run locally in a terminal.

Virtual environment

Create a python 3.9 virtual environment using (please ensure you have [miniconda installed](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html)):

```
conda create -n myenv python=3.9
```

Then activate virtual environemt:

```
conda activate myenv
```

## Add myenv to Jupyter-Notebook/Lab

To ensure python versions are compatible between myenv and Jupyter it is necessary to create myenv IPython kernel.

Begin by installing `ipykernel`:

```
pip install --user ipykernel
```

Then link the myvenv kernel to Jupyter:

```
python -m ipykernel install --user --name=myenv
```

After running Jupyter select `Kernel` from the Jupyter menu bar and select `Change kernel...` from the Kernel menu. From the pop up box select the `myenv` kernel.

## Install dependencies

To install dependencies use the below command:

```
pip install -r requirements.txt
```

## Data

The dataset is an unstructured assortment of ProjectPro customer service enquiry chat logs. The chat logs consist of timestamped dialogue between a human customer agent and a visitor to the ProjectPro website. The dialogue consists predominately of queries about ProjectPro's services, prices, location, and signup information.

## Preprocess, explore, and cluster

Initialise the Cluster class to begin these steps:

```
python engine.py --cluster
```

### Preprocessing

Data label clustering is performed in an unsupervised way. An initial step before any clustering is to preprocess the chat logs one by one:

* Extract only the text transcripts with the relevant chatter.
* Remove urls.
* Normalise contractions and other shorthand.
* Strip everything except letter characters.
* lemmatize words.
* load text and user into dataframe

### Exploratory Data Analysis

Following preprocessing, the data is then explored to identify and vidualise features. Beginning with initialising the EDA object:

```
python engine.py --eda
```

Various data exploration methods can be called on the data to explore the features:

```
# check token frequency distribution
python engine.py --eda_token_dist

# plot the frequency distribution
python engine.py --eda_plot_token_dist

# get top N tokens
python engine.py --eda_top_n_tokens N  # integer

# get token length histogram
python engine.py --eda_tokens_hist

# get senth length histogram
python engine.py --eda_sent_hist
```

The dataframe derived from preprocessing is clustered using the [chatintents](https://github.com/dborrelli/chat-intents) module. The follwing is the order in which the data is clustered:

* The utterances are embedded using Google's [Universal Sentence Encoder](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46808.pdf).
* Hyperparameter tuning is performed using bayesian optimisation, which performs better than random.
* The dimensionality is reduced using UMAP due to [the effects of high dimensionality on distance metrics](https://bib.dbvis.de/uploadedFiles/155.pdf).
* The data is then clustered using HDBSCAN.
* Finally using using spaCy dependency parsing, the data is automatically labelled.

Along the way the outcomes of different stages in the clustering process can be eplored using the following methods:

```
# check the best hyperparameters derived through bayesian optimization
python engine.py --cluster_best_params

# get cluster visualisation
python engine.py --cluster_plot

# get summary dataframe slice (20) of cluster labels e.g. label count
python engine.py --cluster_labels_summary

# get dataframe slice (20) of labelled text
python engine.py --cluster_labeled_utts
```

The final dataframe is exported to a CSV file for further human review and ammendment. The data can be then exported to a json file or kept in the csv and processed using `parse_data_csv` function.

## Train the model

To train the model, run:

```
python engine.py --train
```

The data is prepared by one hot encoding the labels and splitting into train, test, eval sets.

Training entails a pre-step of hyperparameter tuning using `keras_tuner`. Hyperparameters such as:

* Number of layers
* Number of perceptrons
* Dropout layer value
* Activation function
* Learning rate
* Number of epochs

Once these are optimized the best hyperparameter configuration is used to train the model.

The mddel has an early stopping mechanism, which uses the validation loss as a stopping condition. Once the validation loss drops, the training continues for a set number of epochs and stops if there is no improvement over the historic best value.

The pre-training process can be explored using a number of methods:

```
# Check the summary of the hyperparameter tuning
python engine.py --train_search_summary

# Check the results of the hyperparameter tuning
python engine.py --train_results_summary

# check the summary of the model
python engine.py --train_model_summary

# get diagram of model
python engine.py --train_model_diagram
```

Once the model has finished training, the model with the best weights is saved.

## Evaluate the model

The saved model can then be evaluated using the test data. To evaluate the model, begin by initializing the Eval class:

```
python engine.py --eval
```

A number of evaluation methods can be called to explore the model performance:

```
# check test data loss and accuracy
python engine.py --eval_test_loss_acc

# Plot of validation vs train accuracy over epochs
python engine.py --eval_acc_plot

# Plot of validation vs train loss over epochs
python engine.py --eval_loss_plot

# Compare predicted vs actual intents of test data
python engine.py --eval_comp_preds

# Get f-score of predicted intents
python engine.py --eval_fscore

# Get confusion matrix of predicted vs actual intents
python engine.py --eval_conf_matrix
```

## Predictions in chat

To run the model in a chat environment use:

```
python engine.py --chat
```

The chat will run in a terminal and simulate a deployed chatbot with predicted responses given some user input.
