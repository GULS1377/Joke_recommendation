import pandas as pd
import os
import spacy


def read_ratings():
    file_name = os.path.join('/Users/abe/Desktop/joke_recomm_sys_mlp/dataset/jester_dataset_2+',
                             'jesterfinal151cols.csv')
    ratings = pd.read_csv(file_name)
    ratings = ratings.drop(ratings.columns[0], axis=1)
    ratings_list = ratings.values.tolist()
    ratings_list = ratings_list[:500]
    return ratings_list


def read_text():
    file_path = "/Users/abe/Desktop/joke_recomm_sys_mlp/dataset/jester_dataset_2/joke_text.txt"
    f = open(file_path, 'r')
    jokes = f.readlines()
    for i in range(150):
        jokes[i] = jokes[i][0:len(jokes[i]) - 1]
    return jokes


def tf_idf(data, jokes_sim):  # (joke text, jokes similarity matrix)
    new_jokes_sim = jokes_sim
    nlp = spacy.load('en_core_web_sm')
    for i in range(150):
        print(i)
        for j in range(150):
            if i == j:
                new_jokes_sim[i][j] = 1
            else:
                doc1 = nlp(data[i])
                doc2 = nlp(data[j])
                new_jokes_sim[i][j] = doc1.similarity(doc2)
            print(j)
            print(new_jokes_sim[i][j])
    return new_jokes_sim


def predict(test_data, jokes_sim, ratings_with_sim):  # ratings, jokes similarity matrix, ratings with similarity matrix
    # for i in range(len(test_data)):
    for i in range(500):
        for j in range(150):
            if test_data[i][j] != 99:
                count1 = 0
                count2 = 0
                for k in range(150):
                    if test_data[i][k] != 99 and jokes_sim[j][k] > 0 and j != k:
                        count1 += jokes_sim[j][k] * test_data[i][k]
                        count2 += abs(jokes_sim[j][k])
                if count2 == 0:
                    ratings_with_sim[i][j] = 0
                    # print(test_data[i][j], jokes_sim[i][j])
                    continue
                ratings_with_sim[i][j] = count1 / count2
                # print(test_data[i][j], jokes_sim[i][j])
    # print("text predict finish!")
    return ratings_with_sim
