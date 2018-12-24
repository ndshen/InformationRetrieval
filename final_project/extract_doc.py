import sys, os
import argparse
import random
from pymongo import MongoClient
import pandas as pd

parser = argparse.ArgumentParser(description="A script that generate random training documents for project members.")
parser.add_argument('password', help="The password of the database server.")
parser.add_argument('name', help="The name of the project member. ex: kay")
parser.add_argument('remainder', type=int, help="The remainder should be one of these: [0, 1, 2, 3]")
parser.add_argument('--size', default=1000, type=int, help="Size of the training documents")
arguments = vars(parser.parse_args())

HOST = '140.112.107.203'
PORT = 27020
USERNAME = 'rootNinja'
PASSWORD = arguments["password"]
DBNAME = 'CrawlHatePolitics_formal'

SIZE = arguments["size"]
name = arguments["name"]
remainder = arguments["remainder"]

dataDir = os.path.join(os.path.dirname(__file__), "training")
userDir = os.path.join(dataDir, name)
outputCSV=os.path.join(userDir, "training_{}.csv".format(name))

output_format = """\
Document Count: {count}
Article id: {id}
Time: {time}
Author: {author}
URL: {url}
Title: {title}
Content: {content}
================================================================================


"""

def connect_to_DB():
    """Connect to mongo database and return the database"""
    client = MongoClient()
    client=MongoClient(host=HOST,port=PORT,username=USERNAME,password=PASSWORD)
    db=client[DBNAME]
    return(db)

def generate_random_sample(seed:str, min:int, max:int, size:int):
    """Generate random non duplicate numbers between min and max"""
    random.seed(seed.lower())
    sample = random.sample([n for n in range(1, max+1)], size)
    return(sorted(sample))

def output_articles(collection, doc_index_list:list, doc_per_file:int=100):
    count = 0
    next_id = doc_index_list[count]
    cursor = collection.find(no_cursor_timeout=True, batch_size=30)
    text_file = ""
    for i, doc in enumerate(cursor):
        if i != next_id:
            # print(i)
            continue

        count += 1
        text = output_format.format(count=count, id=i, time=doc["Time"], author=doc["Author"], url=doc["URL"], title=doc["ArticleName"],content=doc["Content"])
        text_file += str(text)

        if (count % doc_per_file) == 0:
            
            with open(os.path.join(userDir, "{}{}.txt".format(name, int(count/doc_per_file))), "w", encoding="utf-8") as f:
                f.write(text_file)
            # except UnicodeEncodeError as err:
            #     print(err)
            #     with open("{}.txt".format(int(count/doc_per_file)), "w") as f:
            #         f.write(text_file.encode("utf8").decode("cp950", "ignore"))
                text_file = ""

        print("Done: {}".format(count), end='\r')

        if count == len(doc_index_list):
            break
        next_id = doc_index_list[count]
            
def output_doc_list_csv(doc_index_list:list, outputFile=outputCSV):
    col1 = {"doc_index":doc_index_list}
    df = pd.DataFrame(col1)
    df.to_csv(outputFile, index=0)

if __name__ == "__main__":
    db = connect_to_DB()
    quarter = int(db["Article"].count()/4) - 1
    sample = generate_random_sample(name, 1, quarter, SIZE)
    doc_index_list = [n*4+remainder for n in sample]
    print(doc_index_list)
    output_articles(db["Article"], doc_index_list, doc_per_file = 100)
    output_doc_list_csv(doc_index_list)