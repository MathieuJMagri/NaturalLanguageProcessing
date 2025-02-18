# %%
#Imports for the assignment
import numpy as np
import pandas as pd
import sklearn
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import GridSearchCV

# %%
#IMPORTANT: WHEN TRAINING A MODEL, RESTARTING THE KERNEL IS A MUST TO NOT OVERFIT. THIS ALSO AVOIDS DATA LEAKAGE BETWEEN THE TRAINING DATA AND THE TESTING DATA

#Importing the facts.txt and the fakes.txt files to create our dataset 

# Turn fact.txt file into pandas DataFrame
dfFacts = pd.read_table("facts.txt", header=None)

#Add last column and fill it with value of fact
dfFacts.insert(0, "Category", "fact")

#Add column names
dfFacts.columns = ['Category', 'Text']
#print(dfFacts)

# Turn fact.txt file into pandas DataFrame
dfFakes = pd.read_table("fakes.txt", header=None)

#Add last column and fill it with value of fact
dfFakes.insert(0, "Category", "fake")

#Add column names
dfFakes.columns = ['Category', 'Text']
#print(dfFakes)

# %%
#Making one global dataframe with both facts and fakes together (this will make the data manipulation easier)
dataframes = [dfFacts, dfFakes]
df = pd.concat(dataframes, ignore_index=True)

#The following code is used to randomize are dataset now.
df = df.sample(n=len(df))
print(df)

# %%
#Before doing any testing on hyperparameters and any preprocessing, we will train our models and then test on a basic 80%/20% data split to compare 
#the models when no preprocessing is done.

#Input Values for the model
X = df['Text']

#Expected output
y = df['Category']

#Create the 80/20 split for training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)


# %%
#Create a pipeline for each model (we will first use Count Vectorizer, and thenTfidVectorizer)
pipelineBNB = Pipeline([("vectorize", CountVectorizer()), ("classifier", BernoulliNB())])
pipelineSVM = Pipeline([("vectorize", CountVectorizer()), ("classifier", LinearSVC())])
pipelineLR = Pipeline([("vectorize", CountVectorizer()), ("classifier", LogisticRegression())])

#Train the models
pipelineBNB.fit(X_train, y_train)
pipelineSVM.fit(X_train, y_train)
pipelineLR.fit(X_train, y_train)

#Test the models
predictionBNB = pipelineBNB.predict(X_test)
predictionSVM = pipelineSVM.predict(X_test)
predictionLR = pipelineLR.predict(X_test)

#Accuracy of models
print(f"Accuracy of BNB model: {accuracy_score(y_test, predictionBNB)}")
print(f"Accuracy of SVM model: {accuracy_score(y_test, predictionSVM)}")
print(f"Accuracy of LR model: {accuracy_score(y_test, predictionLR)}")



# %%
#We will now to the same process for the TfidVectorizer

pipelineBNB = Pipeline([("vectorize", TfidfVectorizer()), ("classifier", BernoulliNB())])
pipelineSVM = Pipeline([("vectorize", TfidfVectorizer()), ("classifier", LinearSVC())])
pipelineLR = Pipeline([("vectorize", TfidfVectorizer()), ("classifier", LogisticRegression())])

#Train the models
pipelineBNB.fit(X_train, y_train)
pipelineSVM.fit(X_train, y_train)
pipelineLR.fit(X_train, y_train)

#Test the models
predictionBNB = pipelineBNB.predict(X_test)
predictionSVM = pipelineSVM.predict(X_test)
predictionLR = pipelineLR.predict(X_test)

#Accuracy of models
print(f"Accuracy of BNB model: {accuracy_score(y_test, predictionBNB)}")
print(f"Accuracy of SVM model: {accuracy_score(y_test, predictionSVM)}")
print(f"Accuracy of LR model: {accuracy_score(y_test, predictionLR)}")



# %%
#The accuracy results have increased significant amounts except for the BNB model simply by switching the vectorizing process after the first attempt
#(LR by 5% and SVM by 10%)
#We will continue to use TfidVectorizer for the rest of the tests given it's more accurate results.

#We will now compare both Lemmatization and Stemming.
dfLemma = df
dfStem = df

#We now Lemmatize the dataset
#THIS HELPER FUNCTION IS NOT MINE, I USED THE FOLLOWING CODE SNIPPET TO HELP WITH LEMMETIZATION: https://www.datasnips.com/90/lemmatise-dataframe-text-using-nltk/
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word, pos="v") for word in words]
    return ' '.join(words)

dfLemma['Text'] = dfLemma['Text'].apply(lemmatize_words)
print(dfLemma)


#Stemming the words in the dataset (inspired by the lemmatization function above)
stemmer = SnowballStemmer(language='english')
def stem_words(text):
    words = text.split()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)
dfStem['Text'] = dfStem['Text'].apply(stem_words)
print(dfStem)


# %%
#We recreate the pipleine the same as before but with the lemmatized data
#Input Values for the Lemmatized data
X = dfLemma['Text']

#display(dfLemma)

#Expected output
y = dfLemma['Category']

#Create the 80/20 split for training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)


pipelineBNB = Pipeline([("vectorize", TfidfVectorizer()), ("classifier", BernoulliNB())])
pipelineSVM = Pipeline([("vectorize", TfidfVectorizer()), ("classifier", LinearSVC())])
pipelineLR = Pipeline([("vectorize", TfidfVectorizer()), ("classifier", LogisticRegression())])

#Train the models
pipelineBNB.fit(X_train, y_train)
pipelineSVM.fit(X_train, y_train)
pipelineLR.fit(X_train, y_train)

#Test the models
predictionBNB = pipelineBNB.predict(X_test)
predictionSVM = pipelineSVM.predict(X_test)
predictionLR = pipelineLR.predict(X_test)

#Accuracy of models
print(f"Accuracy of BNB model with Lemmatization: {accuracy_score(y_test, predictionBNB)}")
print(f"Accuracy of SVM model with Lemmatization: {accuracy_score(y_test, predictionSVM)}")
print(f"Accuracy of LR model with Lemmatization: {accuracy_score(y_test, predictionLR)}")


# %%
#We recreate the pipleine the same as before but with the stemmed data
#Input Values for the stemmed data
X = dfStem['Text']

#print(X)

#Expected output
y = dfStem['Category']

#Create the 80/20 split for training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)


pipelineBNB = Pipeline([("vectorize", TfidfVectorizer()), ("classifier", BernoulliNB())])
pipelineSVM = Pipeline([("vectorize", TfidfVectorizer()), ("classifier", LinearSVC())])
pipelineLR = Pipeline([("vectorize", TfidfVectorizer()), ("classifier", LogisticRegression())])

#Train the models
pipelineBNB.fit(X_train, y_train)
pipelineSVM.fit(X_train, y_train)
pipelineLR.fit(X_train, y_train)

#Test the models
predictionBNB = pipelineBNB.predict(X_test)
predictionSVM = pipelineSVM.predict(X_test)
predictionLR = pipelineLR.predict(X_test)

#Accuracy of models
print(f"Accuracy of BNB model with Lemmatization: {accuracy_score(y_test, predictionBNB)}")
print(f"Accuracy of SVM model with Lemmatization: {accuracy_score(y_test, predictionSVM)}")
print(f"Accuracy of LR model with Lemmatization: {accuracy_score(y_test, predictionLR)}")


# %%
#Stemming and Lemmatization did not do much in terms of accuracy and in some cases it even diminished by a little bit.
#We will now test two other pre-processing techniques without using stemming or lemmatization since they do not add a clear benefit to our accuracy.

#We will test stop_words usage through the TfidfVectorizer directly

pipelineBNB = Pipeline([("vectorize", TfidfVectorizer(stop_words="english")), ("classifier", BernoulliNB())])
pipelineSVM = Pipeline([("vectorize", TfidfVectorizer(stop_words="english")), ("classifier", LinearSVC())])
pipelineLR = Pipeline([("vectorize", TfidfVectorizer(stop_words="english")), ("classifier", LogisticRegression())])

#Train the models
pipelineBNB.fit(X_train, y_train)
pipelineSVM.fit(X_train, y_train)
pipelineLR.fit(X_train, y_train)

#Test the models
predictionBNB = pipelineBNB.predict(X_test)
predictionSVM = pipelineSVM.predict(X_test)
predictionLR = pipelineLR.predict(X_test)

#Accuracy of models
print(f"Accuracy of BNB model with stop-words removed: {accuracy_score(y_test, predictionBNB)}")
print(f"Accuracy of SVM model stop-words removed: {accuracy_score(y_test, predictionSVM)}")
print(f"Accuracy of LR model stop-words removed: {accuracy_score(y_test, predictionLR)}")



# %%
#Removing the Stop-Words produced worse results than without removing the stop-words. Giving that we do not have an extansive dataset, this makes sense as 
#we should not be removing words when our quantity of them is already quite limited.

#We will now test n-grams with n=1 to n= (unigrams to bigrams) and then we will do form n=1 to n=3 (unigrams, bigrams and trigrams)

#n=1 to n=2

pipelineBNB = Pipeline([("vectorize", TfidfVectorizer(ngram_range=(1, 2))), ("classifier", BernoulliNB())])
pipelineSVM = Pipeline([("vectorize", TfidfVectorizer(ngram_range=(1, 2))), ("classifier", LinearSVC())])
pipelineLR = Pipeline([("vectorize", TfidfVectorizer(ngram_range=(1, 2))), ("classifier", LogisticRegression())])

#Train the models
pipelineBNB.fit(X_train, y_train)
pipelineSVM.fit(X_train, y_train)
pipelineLR.fit(X_train, y_train)

#Test the models
predictionBNB = pipelineBNB.predict(X_test)
predictionSVM = pipelineSVM.predict(X_test)
predictionLR = pipelineLR.predict(X_test)

#Accuracy of models
print(f"Accuracy of BNB model with 2-gram: {accuracy_score(y_test, predictionBNB)}")
print(f"Accuracy of SVM model 2-gram: {accuracy_score(y_test, predictionSVM)}")
print(f"Accuracy of LR model 2-gram: {accuracy_score(y_test, predictionLR)}")


# %%
#Using n-grams from n=1 to n=2 seemed to have increased the accuracy for the models.

#n=1 to n=3

pipelineBNB = Pipeline([("vectorize", TfidfVectorizer(ngram_range=(1, 3))), ("classifier", BernoulliNB())])
pipelineSVM = Pipeline([("vectorize", TfidfVectorizer(ngram_range=(1, 3))), ("classifier", LinearSVC())])
pipelineLR = Pipeline([("vectorize", TfidfVectorizer(ngram_range=(1, 3))), ("classifier", LogisticRegression())])

#Train the models
pipelineBNB.fit(X_train, y_train)
pipelineSVM.fit(X_train, y_train)
pipelineLR.fit(X_train, y_train)

#Test the models
predictionBNB = pipelineBNB.predict(X_test)
predictionSVM = pipelineSVM.predict(X_test)
predictionLR = pipelineLR.predict(X_test)

#Accuracy of models
print(f"Accuracy of BNB model with 3-gram: {accuracy_score(y_test, predictionBNB)}")
print(f"Accuracy of SVM model 3-gram: {accuracy_score(y_test, predictionSVM)}")
print(f"Accuracy of LR model 3-gram: {accuracy_score(y_test, predictionLR)}")


# %%
#Training with n=1 to n=3 drastically reduced the accuracy of the BNB model, without increasing the accuracy of the other models.
#We will therefore reamin with n=1 to n=2 for the n-grams preprocessing parameter.

# %%
#We will now train the hyperparameters of each model while doing 5-Fold Cross Validation which means each fold will constitue 20% of our data. 
# The reason for using cross-validation is simply because our corpus of information is quite small (400 entries)
# We will use GridSearchCV to accomplish our hyperparameter tuning for all models.

#We start with the tuning of the SVM
pipelineSVM = Pipeline([("vectorize", TfidfVectorizer()), ("classifier", LinearSVC())])

#Defining parameters 
parametersLSVC = [{'classifier__C': [0.1, 1, 10, 100],  
              'classifier__loss': ['hinge', 'squared_hinge'], 
              'classifier__penalty':['l1', 'l2']}]

LSVC_CV = GridSearchCV(pipelineSVM, param_grid=parametersLSVC, scoring='accuracy', cv=5)

#This will give us the best training parameters
LSVC_CV.fit(X_train, y_train)
print("Best Parameters: ")
print(LSVC_CV.best_params_)

#Test on Test Set
predictions = LSVC_CV.predict(X_test) 
print("Accuracy Score: ")
print(accuracy_score(y_test, predictions))

# %%
#We now do the tuning of the Naive Bayes model (Bernouilli)
pipelineBNB = Pipeline([("vectorize", TfidfVectorizer()), ("classifier", BernoulliNB())])

#Defining Parameters
parametersBNB = [{'classifier__alpha': [0.1, 1, 10, 100],  
              'classifier__fit_prior': [True, False]}]

BNB_CV = GridSearchCV(pipelineBNB, param_grid=parametersBNB, scoring='accuracy', cv=5)

#This will give us the best training parameters
BNB_CV.fit(X_train, y_train)
print("Best Parameters: ")
print(BNB_CV.best_params_)

#Test on Test Set
predictions = BNB_CV.predict(X_test) 
print("Accuracy Score: ")
print(accuracy_score(y_test, predictions))


# %%
#We start with the tuning of the SVM
pipelineLR = Pipeline([("vectorize", TfidfVectorizer()), ("classifier", LogisticRegression())])

#Defining Parameters
parametersLR = [{'classifier__C': [0.1, 1, 10, 100],  
              'classifier__penalty':['l1', 'l2', 'elasticnet']}]

LR_CV = GridSearchCV(pipelineLR, param_grid=parametersLR, scoring='accuracy', cv=5)

#This will give us the best training parameters
LR_CV.fit(X_train, y_train)
print("Best Parameters: ")
print(LR_CV.best_params_)

#Test on Test Set
predictions = LR_CV.predict(X_test) 
print("Accuracy Score: ")
print(accuracy_score(y_test, predictions))

# %%
#After fien tuning, each model had an increase in accuracy besides linear SVM. This could be because the default parameters of the linear SVM might be similar to the best 
#parameters after fine-tuning.

#Now, we will do one final experiement where we will take the best performing model, and combine it with the best pre-processing techniques.

#The best model after fine-tuning was the Bernouilli Naive Bayes with parameters {'classifier__alpha': 0.1, 'classifier__fit_prior': True}. 
#We will ignore stemming, lemmatizing, and removing stop-words giving that they did not influence the accuracy.
#We will include bingrams from n=1 to n=2 since it did influence postively our accuracy.

#Create a pipeline for each model thenTfidVectorizer
pipelineBNBFinal = Pipeline([("vectorize", TfidfVectorizer(ngram_range=(1, 2))), ("classifier", BernoulliNB(alpha=0.1, fit_prior=True))])


#Train the model
pipelineBNBFinal.fit(X_train, y_train)


#Test the models
predictionBNBFinal = pipelineBNBFinal.predict(X_test)

#Accuracy of models
print(f"Accuracy of the final BNB model: {accuracy_score(y_test, predictionBNBFinal)}")


# %%
#The accuracy actually increased minimally.

#Limitations of the study: The biggest limit of this study is the fact that our dataset is too small. To be able to detect whether something is a fact or not in a more
#accurate fashion, we would need to have tons more data. Furthermore, fine-tuning the hypeparameters would need to go hand in hand with preprocessing techinques,
# #to be able to get, not only the best possible parameters for a given model, but to also get the best combination of hyperparameters and preprocessing techniques
# #used together.
# 
# Speculates on the generalizability of the results of this study: I do not believe it to be very generalizable given that our dataset is extremely small 
# and facts about Montréal were the only facts used. Indeed, to create something that would be able to be generalized, we would need tons more data on many more cities. 
# Additionally, our data comes from a generative AI model, and we’ve assumed that everything is has generated for us is correct, when in reality, 
# the model may have supplied inaccurate facts, which can lead to training a model with bad data

#****This python file was converted from a .ipynb  file so it may need to be run in an interactive window through VSCode to work


