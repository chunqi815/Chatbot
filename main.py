# Import Python libraries
import random
import string
import datetime as dt

import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer  # ability for lemmatisation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")  # required package for tokenisation
nltk.download("wordnet")  # word database
nltk.download("universal_tagset")
nltk.download("omw-1.4")

# Reading the Corpus of Question Excel
dataset = pd.read_csv("COMP3071-Dataset.csv")
questionDataset = list(dataset['Question'])
questionDataset = ' '.join(str(e) for e in questionDataset).lower()
answerDataset = list(dataset['Answer'])
answerDataset = ' '.join(str(e) for e in answerDataset).lower()

# Reading the Corpus of Small Talk Excel
smallTalkdataset = pd.read_csv("smallTalk.csv")
smallTalkQuesDataset = list(dataset['Question'])
smallTalkQuesDataset = ' '.join(str(e) for e in smallTalkQuesDataset).lower()
smallTalkAnswerDataset = list(dataset['Answer'])
smallTalkAnswerDataset = ' '.join(str(e) for e in smallTalkAnswerDataset).lower()


# Data text pre-processing
def token_lemmatisation(text):
    # remove punctuation
    remove_punctuation = dict((ord(punct), None) for punct in string.punctuation)

    sentence = text.lower().translate(remove_punctuation)
    sentence_token = word_tokenize(sentence)

    partOfSpeech = {
        'ADJ': 'a',
        'ADV': 'r',
        'NOUN': 'n',
        'VERB': 'v'
    }

    # process the lemmatisation with tags
    post = nltk.pos_tag(sentence_token, tagset='universal')

    # part of speech annotation
    lemmatiser = WordNetLemmatizer()
    new_tokens = []

    for token in post:
        word, tag = token[0], token[1]
        if tag in partOfSpeech.keys():
            new_tokens.append(lemmatiser.lemmatize(word, partOfSpeech[tag]))
        else:
            new_tokens.append(lemmatiser.lemmatize(word))

    return new_tokens


# similarity function
def similarity_function(query, token):
    TfidfVec = TfidfVectorizer(tokenizer=token_lemmatisation, min_df=0.01)
    tfidf = TfidfVec.fit_transform(token).toarray()
    tfidf_query = TfidfVec.transform([query]).toarray()
    vals = cosine_similarity(tfidf_query, tfidf)
    return vals


# Generating response for question answering
def question_respond(user_response, token):
    TfidfVec = TfidfVectorizer(tokenizer=token_lemmatisation)
    tfidf = TfidfVec.fit_transform(token)
    tfidf_query = TfidfVec.transform([user_response]).toarray()
    vals = cosine_similarity(tfidf_query, tfidf)
    idx = np.argmax(vals)

    if vals.max() > 0.5:
        return dataset['Answer'][idx]
    else:
        return "I am sorry, unable to understand you. Can you please repeat your question again. "


# Generating response for small talk
def smalltalk_respond(user_response, token):
    TfidfVec = TfidfVectorizer(tokenizer=token_lemmatisation)
    tfidf = TfidfVec.fit_transform(token)
    tfidf_query = TfidfVec.transform([user_response]).toarray()
    vals = cosine_similarity(tfidf_query, tfidf)
    idx = np.argmax(vals)

    if vals.max() > 0:
        return smallTalkdataset['Answer'][idx]
    else:
        return None


# Generating response for date time
def datetime_response(str):
    date = dt.datetime.now()
    if str == 'time':
        hour = date.strftime("%H")
        minute = date.strftime("%M")
        second = date.strftime("%S")
        print("Botty: The time is %s:%s:%s now. " % (hour, minute, second))

    else:
        year = date.year
        month = date.month
        day = date.day
        print("Botty: Today is %s/%s/%s!" % (day, month, year))


# Greetings matching
greetings_inputs = ["hello", "hi", "what's up", "how are you", "greetings", "sup"]
greetings_responses = ["Hi", "Hey", "Hey there", "Hi there", "Hello"]

# Question related to name
name_inputs = ["What is my name", "Do you know who I am", ]


# Intent Routing
def intent_matching(user_response):

    answer_value = similarity_function(user_response, dataset['Question'].tolist()).max()
    smalltalk_answer_value = similarity_function(user_response, smallTalkdataset['Question'].tolist()).max()
    greeting_value = similarity_function(user_response, greetings_inputs).max()

    val_arr = [answer_value, smalltalk_answer_value, greeting_value]
    if max(val_arr) == 0:
        return "I am sorry, unable to understand you. Can you please repeat your question again. "
    else:
        idx = np.argsort(val_arr, None)[-1]

        if idx == 0:
            return question_respond(user_response, dataset['Question'])
        elif idx == 1:
            return smalltalk_respond(user_response, smallTalkdataset['Question'])
        else:
            return random.choice(greetings_responses)


# Defining the chatflow
# identity management
print('')
print("Botty: Hi, I'm Botty. May I know what's your name?")
nameless = True
while nameless == True:
    name = input()
    if name == "":
        print("Botty: I'm sorry. I didn't get your name, can you reenter your name?")
        print('')
    else:
        nameless = False

# chatbot interface
flag = True
print('')
print(
    "Botty: Hello " + name + ". My name is Botty, your personal friendly chatbot. Feel free to ask me question or "
                             "type Bye if you want to leave the conversation.")

while flag == True:
    user_response = input(name + ": ")
    user_response = user_response.lower()
    if user_response != "bye":
        if 'time' in user_response:
            print('')
            datetime_response('time')
            print('')
            print("Botty: So " + name + ", what else may I help you? If you want to exit, type Bye!")
        elif 'today' in user_response or 'date' in user_response:
            print('')
            datetime_response('today')
            print('')
            print("Botty: So " + name + ", what else may I help you? If you want to exit, type Bye!")
        elif user_response == "what is my name":
            print('')
            print("Botty: Your name is " + name)
            print('')
            print("Botty: So " + name + ", what else may I help you? If you want to exit, type Bye!")
        else:
            print('')
            print("Botty: ", end="")
            print(intent_matching(user_response))
            print('')
            print("Botty: So " + name + ", what else may I help you? If you want to exit, type Bye!")
    else:
        flag = False
        print('')
        print("Botty: Bye bye " + name + "! Have a great day.")
