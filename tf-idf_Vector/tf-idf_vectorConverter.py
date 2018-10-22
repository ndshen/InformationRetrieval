import os
import re
import json
import math
from collections import defaultdict
import nltk
from nltk.corpus import stopwords

def extract_terms(text_file):
    # Tokenization
    # parse out all the spaces and some specific punctuation (not including hyphens)
    token_list = re.split(r'[\s\.,:!?;\[\]{}\"\`\'\(\)/]+', text_file)
    
    # Lowercasing everything
    for i, token in enumerate(token_list):
        token_list[i] = token.lower()

    # Stemming using Porter’s algorithm
    stemmer = nltk.stem.PorterStemmer()
    for i, token in enumerate(token_list):
        token_list[i] = stemmer.stem(token)

    # Stopword Removal
    final_token_list = []
    stop_words = set(stopwords.words('english'))
    for token in token_list:
        if token not in stop_words and token != '':
            final_token_list.append(token)
    
    return(final_token_list)

def construct_dictionary(dir_path):
    # check if the argument dir_path is valid
    if os.path.isdir(dir_path) is not True:
        print("Cannot find the documents directory: {}".format(dir_path))
        return(False)
    
    dictionary = defaultdict(dict)
    documents_list = os.listdir(dir_path)
    # iterate through each document
    for doc in documents_list:
        f = open(os.path.join(dir_path, doc), "r")
        text = f.read()
        f.close()
        # to avoid duplicate terms, turn the list into set 
        terms_set = set(extract_terms(text))
        for term in terms_set:
            if term not in dictionary:
                dictionary[term]["df"] = 1
                dictionary[term]["term_id"] = len(dictionary)
                # for debug use
                # dictionary[term]["document_id"] = doc
            else:
                dictionary[term]["df"] += 1

    # # To meet the output file requirement, this method will not be used, though I think it is the more elegant way.
    # with open(os.path.join(os.path.dirname(__file__), "dictionary_unformal.txt"), "w") as f:
    #     f.write(json.dumps(dictionary))

    # This is the output format the the assignment ask for
    with open(os.path.join(os.path.dirname(__file__), "dictionary.txt"), "w") as f:
        f.write("{:<7}{:>20}{:>6}\n".format("t_index", "term", "df"))
        for key, value in dictionary.items():
            f.write("{:<6}{:>20}{:>6}\n".format(value["term_id"], key, str(value["df"])))

def compute_tfIdf(tf, df, N):
    idf = math.log(N/df)
    return(round(tf*idf, 3))

def doc_to_vector(document_file, dictionary_file, output_file, all_document_num):
        f = open(document_file, "r")
        terms_list = extract_terms(f.read())
        f.close()

        # load the dictionary
        dictionary = defaultdict(dict)
        with open(dictionary_file, "r") as d:
            lines = d.read().split('\n')
            for line in lines[1:-1]:
                elements = re.split(r'\s+', line)
                dictionary[elements[1]]["id"] = elements[0]
                dictionary[elements[1]]["df"] = elements[2]

        # find the tf of the terms
        term_table = defaultdict(dict)
        for term in terms_list:
            if term not in term_table:
                term_table[term]["tf"] = 1
                term_table[term]["df"] = int(dictionary[term]["df"])
                term_table[term]["id"] = int(dictionary[term]["id"])
            else:
                term_table[term]["tf"] += 1
        # print(sorted(term_table.items(), key = lambda x : x[1]["id"], reverse=False))
        with open(output_file, "w") as outputF:
            outputF.write("{} {}\n".format("t_index", "tf-idf"))
            for entry in sorted(term_table.items(), key = lambda x : x[1]["id"], reverse=False):
                outputF.write("{:>7} {:>6}\n".format(entry[1]["id"], compute_tfIdf(entry[1]["tf"], entry[1]["df"], all_document_num)))

def dict_dot_product(dictX, dictY):
    result = 0
    for key, value in dictX.items():
        if key in dictY:
            result += (value*dictY[key])
    return(result)

def cosine(docX, docY):
    dictX = dict()
    dictY = dict()
    with open(docX, "r") as doc_x, open(docY, "r") as doc_y:
        for line in doc_x.read().split('\n')[1:-1]:
            tup = re.split(r'\s+', line)
            dictX[int(tup[1])] = float(tup[2])
        for line in doc_y.read().split('\n')[1:-1]:
            tup = re.split(r'\s+', line)
            dictY[int(tup[1])] = float(tup[2])
    
    dot_product = dict_dot_product(dictX, dictY)
    lengthX = math.sqrt(dict_dot_product(dictX, dictX))
    lengthY = math.sqrt(dict_dot_product(dictY, dictY))

    return dot_product/(lengthX*lengthY)


if __name__ == "__main__":
    construct_dictionary(os.path.join(os.path.dirname(__file__), "IRTM"))
    doc_to_vector("./IRTM/1.txt", "dictionary.txt", "1.txt", 1095)
    doc_to_vector("./IRTM/2.txt", "dictionary.txt", "2.txt", 1095)
    print(cosine('1.txt', '2.txt'))