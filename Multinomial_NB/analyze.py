import os
import class_feature

training_input = os.path.join(os.path.dirname(__file__), "training.txt")

def combine_class_docs():
    classes_docs = class_feature.get_training_classes(inputFile=training_input)
    
    for c, docs in classes_docs.items():
        class_combine_text = ""
        for doc in docs:
            print(c, doc)
            text = "{}\n{}\n\n\n".format(doc, class_feature.fetch_document(doc))
            class_combine_text += text
        
        with open(os.path.join(os.path.dirname(__file__), "{}.txt".format(c)), "w") as f:
            f.write(class_combine_text)

if __name__ == "__main__":
    combine_class_docs()