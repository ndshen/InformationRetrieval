import os
import re
import json
import math
from collections import defaultdict

import nltk
from nltk import sent_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def extract_terms_lemma(text_file):
    """Uses NLTK Lemmatizer module to extract token from a text file"""

    def get_wordnet_pos(treebank_tag):
        """Get the wordnet pos for WordNetLemmatizer"""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None
    
    sentences = sent_tokenize(text_file) #split text_file to list of sentences

    tokens = []
    lemmatizer = WordNetLemmatizer()
    for sentence in sentences:
        for word, pos in pos_tag(word_tokenize(sentence)):
            wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
            tokens.append(lemmatizer.lemmatize(word, pos=wordnet_pos))

    # As long as there is a non-punctuation exist in the word, keep it.
    # It allow words like 'Mr.' or 'run-off' to be in the final token list
    words = [word.lower() for word in tokens if re.search(r'\w+', word)] #filter out punctuation and lower cases

    stop_words = set(stopwords.words('english'))
    custom_stop_words = {'\'s', '\'t', '\'ll', '\'m', 'n\'t'} # ex: cat's, don't
    stop_words = stop_words.union(custom_stop_words) #add custom stop words
    final_token_list = [word for word in words if word not in stop_words and len(word) != 1] #filter out stop words and single letter
    
    return(final_token_list)

def construct_dictionary(dir_path, dictionary_file):
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
        terms_set = set(extract_terms_lemma(text))
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

    # This is the output format that the assignment ask for
    with open(os.path.join(os.path.dirname(__file__), dictionary_file), "w") as f:
        f.write("{:<7}{:>20}{:>6}\n".format("t_index", "term", "df"))
        for key, value in dictionary.items():
            f.write("{:<6}{:>20}{:>6}\n".format(value["term_id"], key, str(value["df"])))

def doc_to_vector(document_file, dictionary_file, output_file, all_document_num):
    def compute_tfIdf(tf, df, N):
        idf = math.log(N/df, 10)
        return(round(tf*idf, 3))

    f = open(document_file, "r")
    terms_list = extract_terms_lemma(f.read())
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

    # compute unit vector for this document
    tf_idf_list = []
    id_list = []
    mag_sum = 0
    for entry in sorted(term_table.items(), key = lambda x : x[1]["id"], reverse=False):
        tf_idf = compute_tfIdf(entry[1]["tf"], entry[1]["df"], all_document_num)
        tf_idf_list.append(tf_idf)
        mag_sum += math.pow(tf_idf, 2)
        id_list.append(entry[1]["id"])
    
    mag_sum = math.sqrt(mag_sum)
    unit_vector = [round(x/mag_sum, 3) for x in tf_idf_list]
    
    with open(output_file, "w") as outputF:
        outputF.write("{} {}\n".format("t_index", "tf-idf"))
        for i in range(len(id_list)):
            outputF.write("{:>7} {:>6}\n".format(id_list[i], unit_vector[i]))

def cosine(docX, docY):
    def dict_dot_product(dictX, dictY):
        result = 0
        for key, value in dictX.items():
            if key in dictY:
                result += (value*dictY[key])
        return(result)

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

    document_dir = os.path.join(os.path.dirname(__file__), "IRTM")
    output_dir = os.path.join(os.path.dirname(__file__), "docVector")
    dictionary_file = os.path.join(os.path.dirname(__file__), "dictionary.txt")
    documents_list = os.listdir(document_dir)
    print("Total documents: {}".format(len(documents_list)))

    construct_dictionary(document_dir, dictionary_file=dictionary_file)

    finished_num = 0
    for doc in documents_list:
        doc_to_vector(os.path.join(document_dir, doc), dictionary_file, os.path.join(output_dir, doc), len(documents_list))
        finished_num += 1
        print("Finish {}".format(finished_num), end='\r')

    print("Cosine Simularity between 1.txt and 2.txt", cosine(os.path.join(output_dir,'1.txt'), os.path.join(output_dir, '2.txt')))

    