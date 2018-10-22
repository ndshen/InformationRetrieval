# TF-IDF Vector   
   
## Description
This is the second assignment of the course, the goal is to:   
1. Construct a dictionary from 1095 English documents, which the dictionary consists document frequency for each term.   
2. A function to convert a document into an unit vector. Each dimension in the vector is the tf-idf value of a term.   
3. A function to compute the cosine simularity of 2 document vectors.   

## Prerequisite
This script uses the function in extract_term.py, make sure you have already follow the steps that set up the environment to run nltk mentioned last time.

## Usage
Place the script under the same directory as folder **IRTM** and **docVector**.   
**IRTM** is the folder containing all the document text files, and **docVector** is the folder where the output files (vector of each document) will be.   

```
python tf-idf_vectorConverter.py
```
Run the script and it should generate **dictionary.txt** and all the tf-idf unit vector files.   
You can use the **__cosine function__** in the script to compute the simularity between two vector files.
