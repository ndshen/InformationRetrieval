# Multinomial_NB  
   
## Description
This is the third assignment of the course, the goal is to implement the Multinomial Naive Bayes document classifier.   
We already have 1095 documents in the IRTM directory, and they are named with the following pattern: 1.txt, 2.txt, 3.txt ...   
In the training.txt file, it shows 13 classes with different amount of training documents behind each classes,   
these are the training data we have, and the rest of the documents (900) in IRTM directory are our testing data.

## Prerequisite
These are the python packages that you will have to install before you can run the scripts.
1. numpy
2. pandas
3. openpyxl
4. itertools
5. nltk

After you download the `nltk` package, run the python shell and type in the following scripts.   
```
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
```

## Usage
Remember to change the path of the files and folders in the beginning of the scripts to point to the actual files in your local computer.
( **IRTM** is the folder containing all the document text files ).

```
python class_feature.py
python classifier.py
```
Run the scripts and it should generate 4 files: **tf_df.xlsx**, **class_feature.xlsx**, **dictionary.xlsx** and **result.txt** .
