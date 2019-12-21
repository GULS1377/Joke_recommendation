# Joke_recommendation

### Description

>  This is the project of joke recommendation system. The dataset is based on Jester.

### Contributor

- Lu Sun: Neighbor-Based Model
- Qing Zhou: Matrix-Factorization Model
- Zihao Ye: Multi-Layer Perceptron Model

### Project Structure

.

|——Multi-Layer Perceptron Model: contains codes of multi-layer perceptron model 

|————model.py: model of mlp

|————preprocessing.py: process the ratings document, and get the train_set and the test_set

|————text_preprocessing.py: process the text part of the jokes and the similarity between different jokes via 'spacy'

|————trainer.py: train the model

|————dataset: containing jester_dataset_2 and jester_data_set_2+

|——————jester_dataset_2: containing ratings and jokes

|————————jester_ratings.dat: ratings of different users on different jokes

|————————joke_text.txt: 150 jokes

|——————jester_dataset_2+: containing ratings and jokes

|————————jesterfinal151cols.csv: ratings of different users on different jokes

|————————joke_text.txt: 150 jokes

|——Neighbor-Based Model: contains codes of neighbor-based model 

|————d2v.model: model of doc2vec

|————data.py: handle training data and test data

|————jesterfinal151cols.xls: dataset

|————joke_text.txt: joke text

|————main.py: main program

|————similarities.py: class of three types of similarities

|————test.xlsx: test dataset

|——Matrix-Factorization Model: contains codes of matrix-factorization based model

|————d2v.model: model of doc2vec

|————raings.csv: dataset

|————joke_text.txt: joke text

|————data: processed data matrix

|————X.npy: processed rating matrix

|————masks.npy: processed mask matrix

|————baseline.py: MF baseline

|————d2v_sim.py: adding d2v info to the model

|——README.md: brief documentation of the whole project