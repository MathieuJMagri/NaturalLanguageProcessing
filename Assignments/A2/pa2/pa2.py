# -*- coding: utf-8 -*-
# %%
#Imports for the assignment
#coding: utf-8
import os
os.system('pip3 install transformers')
os.system('pip3 install torchvision ')
os.system('pip3 install torch torchvision')
os.system('pip3 install torch torchvision torchaudio')
os.system('pip install torch')
os.system('pip3 install numpy')
os.system('pip3 install scipy')
os.system('pip install wheel')
os.system('pip install pandas')

import numpy as np
import pandas as pd
import sklearn
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize, pos_tag
from nltk.wsd import lesk 
from nltk.corpus import wordnet as wn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# %%
#loader.py file used to load the data giving in the assignment handout sheet
'''
@author: jcheung

Developed for Python 2. Automatically converted to Python 3; may result in bugs.
'''
import xml.etree.cElementTree as ET
import codecs

#I CHANGED THE STARTER CODE HERE TO MAKE SURE THE LEMMA AND CONTEXT ARE STRINGS AND NOT BYTES
class WSDInstance:
    def __init__(self, my_id, lemma, context, index):
        self.id = my_id         # id of the WSD instance
        self.lemma = lemma.decode('utf-8') if isinstance(lemma, bytes) else lemma      # lemma of the word whose sense is to be resolved 
        self.context = [word.decode('utf-8') if isinstance(word, bytes) else word for word in context]  # lemma of all the words in the sentential context
        self.index = index      # index of lemma within the context
    def __str__(self):
        '''
        For printing purposes.
        '''
        return '%s\t%s\t%s\t%d' % (self.id, self.lemma, ' '.join(self.context), self.index)

def load_instances(f):
    '''
    Load two lists of cases to perform WSD on. The structure that is returned is a dict, where
    the keys are the ids, and the values are instances of WSDInstance.
    '''
    tree = ET.parse(f)
    root = tree.getroot()
    
    dev_instances = {}
    test_instances = {}
    
    for text in root:
        if text.attrib['id'].startswith('d001'):
            instances = dev_instances
        else:
            instances = test_instances
        for sentence in text:
            # construct sentence context
            context = [to_ascii(el.attrib['lemma']) for el in sentence]
            for i, el in enumerate(sentence):
                if el.tag == 'instance':
                    my_id = el.attrib['id']
                    lemma = to_ascii(el.attrib['lemma'])
                    instances[my_id] = WSDInstance(my_id, lemma, context, i)
    return dev_instances, test_instances

def load_key(f):
    '''
    Load the solutions as dicts.
    Key is the id
    Value is the list of correct sense keys. 
    '''
    dev_key = {}
    test_key = {}
    for line in open(f):
        if len(line) <= 1: continue
        #print (line)
        doc, my_id, sense_key = line.strip().split(' ', 2)
        if doc == 'd001':
            dev_key[my_id] = sense_key.split()
        else:
            test_key[my_id] = sense_key.split()
    return dev_key, test_key

def to_ascii(s):
    # remove all non-ascii characters
    return codecs.encode(s, 'ascii', 'ignore')

if __name__ == '__main__':
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'
    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)
    
    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k:v for (k,v) in dev_instances.items() if k in dev_key}
    test_instances = {k:v for (k,v) in test_instances.items() if k in test_key}
    
    # read to use here
    print(len(dev_instances)) # number of dev instances
    print(len(test_instances)) # number of test instances
    totalDev = len(dev_instances)
    totalTest = len(test_instances)
    

# %%
#Observing the dev_instances and the dev_key instances
print("dev_instances:") 
print(dev_instances)

#dev_key, key = ID of the word; Values = the correct sense of the word
#group%1:03:00:: means the following
#group: This is the lemma or base form of the word.
#%1: This part of the notation indicates the part of speech. Here, "1" stands for "noun."
#03: This is the lexicographer file number, which categorizes the word semantically. "03" refers to nouns related to groups or collections.
#00: This specifies the sense within the lexicographer file, representing the ordering of senses by frequency or usage.
#when you see group%1:03:00::, it does match with WordNet’s group.n.01 (This is synset's definition, word.partofspeech.def)
#Both the lemma (e.g., "group") and the POS tag (e.g., noun) need to match to ensure they refer to the same concept in WordNet.
print("dev_key:") 
print(dev_key)

#id of the WSD instance, lemma is the word needed, context is its context, index of the word in its context
print("Five WSD object")
print(dev_instances.get("d001.s001.t002"))
print(dev_instances.get("d001.s001.t003"))
print(dev_instances.get("d001.s001.t004"))
print(dev_instances.get("d001.s002.t002"))
print(dev_instances.get("d001.s002.t005"))

#wn.synset_from_sense_key("driving%1:04:03::") -> gives a wordnet to sense key like Synset('drive.n.06') in this case
values = list(dev_key.values())
print(values)
print(values[0])
wn.synset_from_sense_key(' '.join(values[0]))



# %%
#Using NLTK's Lesk's algorithm on the devset
#RUNNING ON DEVSET
#Run Lesk's algorithm by iterating over all WSD object instances and creating a list of Synset definitions
dicDevPredictedSynsets = {}
for key, value in dev_instances.items():
    dicDevPredictedSynsets[key] = lesk(dev_instances.get(key).context, dev_instances.get(key).lemma)

#All prediected synsets using NLKT's Lesk algorithm
print(dicDevPredictedSynsets)   

#Now we must compare these synset values to the one's we have in the devset
answersDevSet = {}
keysSynsets = list(dev_key.keys())
for key in keysSynsets:
    value = dev_key.get(key)
    for value in value:
        answersDevSet[key] = wn.synset_from_sense_key(value)


print(answersDevSet)

#Compare both lists
correct_predictionsDev = 0
for key, value in dicDevPredictedSynsets.items():
    if dicDevPredictedSynsets.get(key) == answersDevSet.get(key):
        correct_predictionsDev = correct_predictionsDev + 1

accuracyDev = correct_predictionsDev/totalDev
print(accuracyDev)


# %%
#RUNNING LESK'S ALGORITHM ON THE TEST SET
#Run Lesk's algorithm by iterating over all WSD object instances and creating a list of Synset definitions
dicTestPredictedSynsets = {}
for key, value in test_instances.items():
    dicTestPredictedSynsets[key] = lesk(test_instances.get(key).context, test_instances.get(key).lemma)

#All prediected synsets using NLKT's Lesk algorithm
print(dicTestPredictedSynsets)   

#Now we must compare these synset values to the one's we have in the devset
answersTestSet = {}
keysSynsets = list(test_instances.keys())
for key in keysSynsets:
    value = test_key.get(key)
    for value in value:
        answersTestSet[key] = wn.synset_from_sense_key(value)


print(answersTestSet)

#Compare both lists
correct_predictionsTest = 0
for key, value in dicTestPredictedSynsets.items():
    if dicTestPredictedSynsets.get(key) == answersTestSet.get(key):
        correct_predictionsTest = correct_predictionsTest + 1

accuracyTest = correct_predictionsTest/totalTest
print(accuracyTest)

#As we can see, the algorithm performs better on the test set than on the dev set, but this can be due to a variety of factors such as the fact that
#the dev set is so small.

# %%
#Lesk's algorithm with POS tag from synset on the DevSet

#DevSet tags
POSTagsDevSet = {}
devKeysSynsets = list(dev_key.keys())
for key in devKeysSynsets:
    value = dev_key.get(key)
    for value in value:
        POSTagsDevSet[key] = wn.synset_from_sense_key(value)

print(POSTagsDevSet)
# a = POSTagsDevSet["d001.s001.t002"]
# print(a)

#This is one POS tag
# str(a).split(".")[1]
POSTagsDevSetKeys = list(POSTagsDevSet.keys())
for key in POSTagsDevSetKeys:
    value = POSTagsDevSet.get(key)
    POSTagsDevSet[key] = str(value).split(".")[1]

print(POSTagsDevSet)

#Run Lesk's algorithm by iterating over all WSD object instances and creating a list of Synset definitions
dicDevPredictedSynsets = {}
for key, value in dev_instances.items():
    dicDevPredictedSynsets[key] = lesk(dev_instances.get(key).context, dev_instances.get(key).lemma, POSTagsDevSet.get(key))

#All prediected synsets using NLKT's Lesk algorithm
print(dicDevPredictedSynsets)   

#Now we must compare these synset values to the one's we have in the devset
answersDevSet = {}
keysSynsets = list(dev_key.keys())
for key in keysSynsets:
    value = dev_key.get(key)
    for value in value:
        answersDevSet[key] = wn.synset_from_sense_key(value)


print(answersDevSet)

#Compare both lists
correct_predictionsDev = 0
for key, value in dicDevPredictedSynsets.items():
    if dicDevPredictedSynsets.get(key) == answersDevSet.get(key):
        correct_predictionsDev = correct_predictionsDev + 1

accuracyDev = correct_predictionsDev/totalDev
print(accuracyDev)

#When adding POS tags, the dev set accuracy increases by about 3% !





# %%
#Lesk's algorithm with POS tags on the Test Set
#Lesk's algorithm with POS tag from synset on the DevSet

#TestSet tags
#We use the split function to get the pos tag from the synset value
POSTagsTestSet = {}
testKeysSynsets = list(test_key.keys())
for key in testKeysSynsets:
    value = test_key.get(key)
    for value in value:
        POSTagsTestSet[key] = wn.synset_from_sense_key(value)

print(POSTagsTestSet)
# a = POSTagsTestSet["d002.s001.t001"]
# print(a)

#This is one POS tag
# str(a).split(".")[1]
POSTagsTestSetKeys = list(POSTagsTestSet.keys())
for key in POSTagsTestSetKeys:
    value = POSTagsTestSet.get(key)
    POSTagsTestSet[key] = str(value).split(".")[1]

print(POSTagsTestSet)

#RUNNING LESK'S ALGORITHM ON THE TEST SET WITH POS TAGS
#Run Lesk's algorithm by iterating over all WSD object instances and creating a list of Synset definitions
dicTestPredictedSynsets = {}
for key, value in test_instances.items():
    dicTestPredictedSynsets[key] = lesk(test_instances.get(key).context, test_instances.get(key).lemma, POSTagsTestSet.get(key))

#All prediected synsets using NLKT's Lesk algorithm
print(dicTestPredictedSynsets)   

#Now we must compare these synset values to the one's we have in the devset
answersTestSet = {}
keysSynsets = list(test_instances.keys())
for key in keysSynsets:
    value = test_key.get(key)
    for value in value:
        answersTestSet[key] = wn.synset_from_sense_key(value)


print(answersTestSet)

#Compare both lists
correct_predictionsTest = 0
for key, value in dicTestPredictedSynsets.items():
    if dicTestPredictedSynsets.get(key) == answersTestSet.get(key):
        correct_predictionsTest = correct_predictionsTest + 1

accuracyTest = correct_predictionsTest/totalTest
print(accuracyTest)

#As se can see, the accuracy of the test set increases by about 4% and is still more accurate than the dev set.

# %%
#Calculating the Baseline form the most used WordSense definition

#Baseline calculation on the DevSet
dicDevPredictedBaseline = {}
for key, value in dev_instances.items():
    dicDevPredictedBaseline[key] = wordnet.synsets(dev_instances.get(key).lemma)[0] #Get the most prevalent definition of the lemma

print(dicDevPredictedBaseline)      

#Compare both lists
correct_predictionsDev = 0
for key, value in dicDevPredictedBaseline.items():
    if dicDevPredictedBaseline.get(key) == answersDevSet.get(key):
        correct_predictionsDev = correct_predictionsDev + 1

accuracyDev = correct_predictionsDev/totalDev
print(accuracyDev)

#Baseline on the dev set performs at a rate of 67% which is much better than any Lesk algorothm! This can make sense if we account for the fact that
#the first definition of a word is often the most used and so the percentage here would be higher.

#Baseline calculation on the TestSet
#Baseline calculation on the DevSet
dicTestPredictedBaseline = {}
for key, value in test_instances.items():
    dicTestPredictedBaseline[key] = wordnet.synsets(test_instances.get(key).lemma)[0] #Get the most prevalent definition of the lemma

print(dicTestPredictedBaseline)

#Compare both lists
correct_predictionsTest = 0
for key, value in dicTestPredictedBaseline.items():
    if dicTestPredictedBaseline.get(key) == answersTestSet.get(key):
        correct_predictionsTest = correct_predictionsTest + 1

accuracyTest = correct_predictionsTest/totalTest
print(accuracyTest)

#The test set performs much bette than the previous iterations of Lesk's algorithm, but underperforms the dev set by 5%. Due to the dev sets small size, accuracy
#mesures can be taken with a grain of salt given that the lower amount of data we have, the less generablizable our results can be considered.


# %%
#For one of the additional models, I will implement a Bayes Classifer
#The reason for this choice is that a NB classifier works well with small datasets (like the one we currently have), 
# it is easy to extend with additional linguistic or semantic features, and it is easy to understand and implement
#Also works well with labelled data which is our case here


#To be able to work with a Naive Bayes classifier, I created my own data with chatGPT. I asked ChatGPT to create sentences that feature the work bank and to
#include the definition used for that instance of bank. The following lines of code will be us preparing the data for the classifier.
#The work "bank" was chosen because it is a work with plenty of possible meanings which would allow us to properly test our classifier.
#We can allow do WSD on 1 work, however, in this case.

#Specific word prompt used: "Generate 150 instances of the word bank used in a sentence as well as its wordnet sense."

#Importing the facts.txt and the fakes.txt files to create our dataset 

# Turn fact.txt file into pandas DataFrame
dfContext = pd.read_table("bank.txt", header=None)
dfContext.columns = ['Context']
#print(dfContext)
dfSenses = pd.read_table("bankSenses.txt", header=None)
dfSenses.columns = ['Senses']
#print(dfSenses)

dfContext['Senses'] = dfSenses['Senses'] #Combine both dataframes
#print(dfContext)

#Before doing any testing on hyperparameters and any preprocessing, we will train our models and then test on a basic 80%/20% data split to compare 
#the models when no preprocessing is done.

#Input Values for the model
X = dfContext['Context']

#Expected output
y = dfContext['Senses']

#Create the 80/20 split for training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

#We will first use countVectorize first and then TfidfVectorizer to tokenize sentences
#Create a pipeline for each model (we will first use Count Vectorizer, and thenTfidVectorizer)
pipelineMNBCV = Pipeline([("vectorize", CountVectorizer()), ("classifier", MultinomialNB())])
pipelineMNBTfid = Pipeline([("vectorize", TfidfVectorizer()), ("classifier", MultinomialNB())])

#Train the model
pipelineMNBCV.fit(X_train, y_train)
pipelineMNBTfid.fit(X_train, y_train)

#Test the model
predictionMNBCV = pipelineMNBCV.predict(X_test)
predictionMNBTfid = pipelineMNBTfid.predict(X_test)

#Accuracy of model
# print(f"Accuracy of MNBCV model: {accuracy_score(y_test, predictionMNBCV)}")
# print(f"Accuracy of MNBTfid model: {accuracy_score(y_test, predictionMNBTfid)}")

#CountVectorizer performs better.  Ideally, we would have more data to properly analyze these results.
#50% is however better than Lesk's algorithm but not the most frequent baseline method. Tfid is also better than Lesk's algorithm but not the baseline method.


# %%
#We will continue to use CountVectorizer for the rest of the tests given its better results
dfLemma = dfContext

#We now Lemmatize the dataset (FROM ASSIGNMENT 1) (an additional experiment could have been to consider ngram words as well)
#THIS HELPER FUNCTION IS NOT MINE, I USED THE FOLLOWING CODE SNIPPET TO HELP WITH LEMMETIZATION: https://www.datasnips.com/90/lemmatise-dataframe-text-using-nltk/
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word, pos="v") for word in words]
    return ' '.join(words)

dfLemma['Context'] = dfLemma['Context'].apply(lemmatize_words)

#We recreate the pipleine the same as before but with the lemmatized data
#Input Values for the Lemmatized data
XLem = dfLemma['Context']

#Expected output
yLem = dfLemma['Senses']

#Create the 80/20 split for training and testing data
X_trainLem, X_testLem, y_trainLem, y_testLem = train_test_split(XLem, yLem, test_size=0.2, random_state=11)

#We will first use countVectorize first and then TfidfVectorizer to tokenize sentences
#Create a pipeline for each model (we will first use Count Vectorizer, and thenTfidVectorizer)
pipelineMNBCV = Pipeline([("vectorize", CountVectorizer()), ("classifier", MultinomialNB())])

#Train the model
pipelineMNBCV.fit(X_trainLem, y_trainLem)

#Test the model
predictionMNBCV = pipelineMNBCV.predict(X_testLem)

#Accuracy of model
# print(f"Accuracy of MNBTfid model: {accuracy_score(y_testLem, predictionMNBCV)}")

#By Lemmatizing the context words, our accuracy actually decreased! We will therefore continue without lemmatizing.

# %%
#We will now fine tune some of the parameters to see if we can get better results
from sklearn.model_selection import GridSearchCV
pipelineMNBCV = Pipeline([("vectorize", CountVectorizer()), ("classifier", MultinomialNB())])

#Defining Parameters
parametersMNBC = [{'classifier__alpha': [0.1, 1, 10, 100],  
              'classifier__fit_prior': [True, False]}]

MNBC_CV = GridSearchCV(pipelineMNBCV, param_grid=parametersMNBC, scoring='accuracy', cv=5)

#This will give us the best training parameters
MNBC_CV.fit(X_train, y_train)
print("Best Parameters: ")
print(MNBC_CV.best_params_)

#Test on Test Set
predictions = MNBC_CV.predict(X_test) 
print("Accuracy Score: ")
print(accuracy_score(y_test, predictions))

#We get an accuracy of 0.5 which isn't better than when we do not hyperparametrize. 

# %%
#Using a pretrained LLM model: BERT
#BERT was selected because it would allow me to use the cosine similarity metric seen in class and I wanted to experiment with it.
#It also seems relatively simple to implement and use with decent results
#Import statements
# pip3 install transformers
# pip3 install torchvision 
# pip3 install torch torchvision
# pip3 install torch torchvision torchaudio
# pip install torch
# pip3 install numpy
# pip3 install scipy



from transformers import BertTokenizer, BertModel
import torch


# %%
# Step 1: Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# %%
#This code was inspired by many online ressources that use BERT.
#Sources: https://sujitpal.blogspot.com/2020/11/word-sense-disambiguation-using-bert-as.html
#https://github.com/sujitpal/deeplearning-ai-examples/blob/master/blog_tds_fd905cb22df7_bert_embeddings_wsd.ipynb 

#Basically, we are using the BERT word embeddings to further analyze the relationship bewteen the lemma and its context. It should lead to an increase in performance
#given that it is a pretrained model.


#Here we are encoding a context using BERT, this using the tokenizer and model variables defined above with our BERT model
def encoding(context):
    inputs = tokenizer(context, return_tensors='pt', truncation=True, max_length=128, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1) #Token embeddings are getting pooled

#Getting the definitions of the given word
def getDefinitions(lemma):
    definitions = wn.synsets(lemma)
    return [(definition.name(), definition.definition()) for definition in definitions]

#Here is in many ways the heart of the algorithm, we disambiguate the word with it's context
def disambiguateWord(context, lemma):
    #Encode the context of the target word
    embedding = encoding(context)
    
    #Get meanings
    senseDefinitions = getDefinitions(lemma)
    
    if not senseDefinitions:
        return "No senses found for the target word."
    
    best_sense = None
    max_similarity = -1
    
    #Context is compared with each sense defintion
    for name, definition in senseDefinitions:
        gloss_embedding = encoding(definition)
        similarity = torch.cosine_similarity(embedding, gloss_embedding).item()
        
        if similarity > max_similarity:
            max_similarity = similarity
            best_sense = name
    
    return best_sense


# %%
#Making predictions on the devSet
best_sensesPredDev = {}
for key, value in dev_instances.items():
    lemma = dev_instances.get(key).lemma
    context = ' '.join(dev_instances.get(key).context)
    best_sensesPredDev[key] = disambiguateWord(context, lemma)




# %%
#Getting accuracy on DevSet
#Compare both lists for accuracy prediction
correct_predictionsDevBERT = 0
for key, value in best_sensesPredDev.items():
    answer = answersDevSet.get(key)
    answer = str(answer).split("'")[1]
    if best_sensesPredDev.get(key) == answer:
        correct_predictionsDevBERT = correct_predictionsDevBERT + 1

accuracyDevBERT = correct_predictionsDevBERT/totalDev
print(accuracyDevBERT)

#The accuracy is 0.495 percent, which is not as good as our NB model but better than the Lesk model. It is unfortunately not better than the baseline model either.

# %%
#We will now test the BERT model on the test set
#Making predictions on the testSet
best_sensesPredTest = {}
for key, value in test_instances.items():
    lemma = test_instances.get(key).lemma
    context = ' '.join(test_instances.get(key).context)
    best_sensesPredTest[key] = disambiguateWord(context, lemma)

# %%
#Getting accuracy on TestSet
#Compare both lists for accuracy prediction
correct_predictionsTestBERT = 0
for key, value in best_sensesPredTest.items():
    answer = answersTestSet.get(key)
    answer = str(answer).split("'")[1]
    if best_sensesPredTest.get(key) == answer:
        correct_predictionsTestBERT = correct_predictionsTestBERT + 1

accuracyTestBERT = correct_predictionsTestBERT/totalTest
print(accuracyTestBERT)

#The accuracy is 0.470 percent, which is worse than the results we got from the Dev set.



