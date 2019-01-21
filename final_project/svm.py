from sklearn import svm
from joblib import dump, load
from pymongo import MongoClient
import jieba
import jieba.posseg as pseg
import numpy as np
import math

# Database arguments
HOST = '140.112.107.203'
PORT = 27020
USERNAME = 'rootNinja'
PASSWORD = 'swordtight'
DBNAME = 'IR_final'
collections = ['Boy-Girl', 'C_Chat', 'HatePolitics', 'Movie', 'NBA', 'Stock']

dictionary_file = 'dictionary.txt'
accept_tags = {'nz', 'nt', 'nrt', 'nrfg', 'nr', 'n', 'a'}

train_num = 500
content_length_threshold = 100

def connect_to_DB():
    """Connect to mongo database and return the database"""
    client = MongoClient()
    client=MongoClient(host=HOST,port=PORT,username=USERNAME,password=PASSWORD)
    db=client[DBNAME]
    return(db)

def load_dictionary(txt_file):
    result = []
    with open(txt_file, 'r', encoding="utf-8") as f:
        lines = f.read().split('\n')
        for line in lines:
            if line == '':
                continue
            result.append(line)
    return(result)

def extract_terms(document:str):
    content = document.replace('\n','')
    words = pseg.cut(content)
    final_token_list = [word for word, tag in words if tag in accept_tags and len(word) > 1]
    return(final_token_list)

def doc_to_vector(document:str, dictionary:list, dimension:int):
    token_list = extract_terms(document)
    article_vector = [0 for j in range(dimension)]
    vector_sum = 0
    for token in token_list:
        if token in dictionary:
            article_vector[dictionary.index(token)] += 1
            vector_sum += 1
    if vector_sum == 0:
        normed_article_vector = article_vector
    else:
        magnitude = math.sqrt(sum(map(lambda x: x**2, article_vector)))
        normed_article_vector = list(map(lambda x: x/magnitude, article_vector))
    return(normed_article_vector)

def classify(document:str, dictionary, clf):
    doc_vec = doc_to_vector(document, dictionary, len(dictionary))
    dec = clf.decision_function([doc_vec])
    # print(dec)
    answer = collections[np.argmax(dec.flatten())]
    return(answer)

if __name__ == '__main__':

    db = connect_to_DB()
    dictionary = load_dictionary(dictionary_file)
    model = load('svm_linear2.joblib')
    test_num = 100
    error = 0

    for collection_name in collections:
        c = db[collection_name]
        print("Doing {}".format(collection_name))
        done = 0
        for i, article in enumerate(c.find(no_cursor_timeout=True, batch_size=30).sort("id", -1)):

            if len(article['Content']) > content_length_threshold:
                if classify(article['Content'], dictionary, model) != collection_name:
                    error += 1

                done += 1
                print('Done: {}'.format(done), end='\r')
                if done == test_num:
                    break
    print()
    print("error: {}".format(error))
    print("accuracy: {}".format(error/test_num*6))


