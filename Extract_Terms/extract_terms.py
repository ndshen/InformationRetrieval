import requests
import re
# from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords

def main(text_file):
    # Tokenization
    # parse out all the spaces and some specific punctuation (not including hyphens)
    token_list = re.split(r'[\s\.,:!?;\[\]{}\"\']+', text_file)
    
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

if __name__ == "__main__":
    r = requests.get('https://ceiba.ntu.edu.tw/course/35d27d/content/28.txt')
    ans = main(r.text)
    with open('IR_bo4705021_hw1_result.txt', "w") as f:
        for term in ans:
            f.write(term+'\n')
    print(ans)