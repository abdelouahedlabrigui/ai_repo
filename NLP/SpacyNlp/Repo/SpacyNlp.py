import datetime
import numpy as np
import pandas as pd
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils import *
from transformers import pipeline, set_seed
import json

def truncate_duplicates(text):
    """
    Truncates the last parts of a string that are duplicated.

    Args:
        text: The input string.

    Returns:
        The string with the last duplicated parts truncated, or the original
        string if no duplicates are found.  Returns an empty string if the
        input is empty.
    """

    if not text:  # Handle empty string input
        return ""

    n = len(text)
    for i in range(n - 1, 0, -1):  # Iterate backwards from the second-to-last character
        substring = text[i:]  # Extract the substring from i to the end
        prefix = text[:i]      # The part of the string before the substring

        if substring in prefix:  # Check if the substring is present in the prefix
            first_occurrence = prefix.find(substring)
            return text[:first_occurrence + len(substring)] #Return up to end of first occurrence
            
    return text  # No duplicates found, return the original string

class Translation:
    def __init__(self, text: str, actor: str, response: str):
        self.text = text
        self.actor = actor
        self.response = response
    def translation_generator(self):
        try:
            if (not self.text.strip()) or (not self.actor.strip()) or (not self.response.strip()):
                raise ValueError("Input text is empty!")
            
            text_generation = TextGeneration(text=self.text, actor=self.actor, response=self.response).text_generation()
            translator = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-es")
            outputs = translator(text_generation, clean_up_tokenization_spaces=True, min_length=140)
            data = []
            translation_text = outputs[0]['translation_text']
            data.append({"Text": text_generation, "EsTranslation": translation_text})
            return data
        except Exception as e:
            raise ValueError(f"{e}")

class TextGeneration:
    def __init__(self, text: str, actor: str, response: str):
        self.text = text
        self.actor = actor
        self.response = response
    def text_generation(self):
        try:
            if (not self.text.strip()) or (not self.actor.strip()) or (not self.response.strip()):
                raise ValueError("Input text is empty!")
            generator = pipeline('text-generation')
            prompt = self.text + f" based on perspective of: {self.actor}: " + self.response
            outputs = generator(prompt, max_length=280)
            generated_text = outputs[0]["generated_text"]
            return generated_text
        except Exception as e:
            raise ValueError(f"{e}")


class SpacyNlp:
    def __init__(self, title, searchString, text):
        self.title = title
        self.searchString = searchString
        self.text = text
        self.nlp = spacy.load("en_core_web_sm")
    def EntityRecognition(self):
        try:
            doc = self.nlp(self.text)
            dataList = []
            for sent in doc.sents:
                for ent in sent.ents:
                    data = {    
                        "Title": str(self.title),
                        "SearchString": str(self.searchString),        
                        "Text": str(ent.text),
                        "StartChat": str(ent.start_char),
                        "EndChar": str(ent.end_char),
                        "Label": str(ent.label_), 
                        "CreatedAT": str(datetime.datetime.now().isoformat())                   
                    }
                    dataList.append(data)
            return dataList
        except Exception as e:
            print(f"An error occurred: {e}")
    def NounChunk(self):
        try:
            doc = self.nlp(self.text)
            dataList = []
            for sent in doc.sents:
                for chunk in sent.noun_chunks:
                    data = {   
                        "Title": str(self.title),
                        "SearchString": str(self.searchString),
                        "Text": str(chunk.text),
                        "RootText": str(chunk.root.text),
                        "RootDep": str(chunk.root.dep_),
                        "RootHead": str(chunk.root.head.text),
                        "CreatedAT": str(datetime.datetime.now().isoformat())
                    }
                    dataList.append(data)
            return dataList
        except Exception as e:
            print(f"An error occurred: {e}")
    def Sentiments(self):
        self.script_executed = True
        try:
            doc = self.nlp(self.text)
            sentiment = SentimentIntensityAnalyzer()  
            dataList = [] 
            for sent in doc.sents:   
                dictionary = sentiment.polarity_scores(str(sent))
                positive = str(round(float(dictionary['pos']), 4))
                negative = str(round(float(dictionary['neg']), 4))
                neutral = str(round(float(dictionary['neu']), 4))
                compound = str(round(float(dictionary['compound']), 4))
                data = {   
                    "Sentence": str(sent),
                    "Title": str(self.title),
                    "SearchString": str(self.searchString),
                    "Positive": str(positive),
                    "Negative": str(negative),
                    "Neutral": str(neutral),
                    "Compound": str(compound), 
                    "CreatedAT": str(datetime.datetime.now().isoformat())              
                }
                dataList.append(data)
            return dataList
        except Exception as e:
            print(e)
            self.script_executed = False
            return self.script_executed
        
    def SentimentsQuerying(self, state):
        self.script_executed = True
        try:
            doc = self.nlp(self.text)
            sentiment = SentimentIntensityAnalyzer()
            dataList = []

            max_positive, max_negative, max_neutral = 0, 0, 0
            for sent in doc.sents:
                scores = sentiment.polarity_scores(str(sent))
                pos, neg, neu, comp = scores['pos'], scores['neg'], scores['neu'], scores['compound']

                data = {
                    "Sentence": str(sent),
                    "Title": str(self.title),
                    "SearchString": str(self.searchString),
                    "Positive": round(pos, 4),
                    "Negative": round(neg, 4),
                    "Neutral": round(neu, 4),
                    "Compound": round(comp, 4),
                    "CreatedAT": datetime.datetime.now().isoformat()
                }
                dataList.append(data)

                # Track max values
                max_positive = max(max_positive, pos)
                max_negative = max(max_negative, neg)
                max_neutral = max(max_neutral, neu)

            # Append only the highest sentiment sentence based on `state`
            if state == "Positive":
                dataList = [d for d in dataList if d["Positive"] == max_positive]
            elif state == "Negative":
                dataList = [d for d in dataList if d["Negative"] == max_negative]
            elif state == "Neutral":
                dataList = [d for d in dataList if d["Neutral"] == max_neutral]
            df = pd.DataFrame(dataList)
            cleanList = []
            for sentence, title, search, pos_, neg_, neu_, com_, createdAT in zip(
                list(df['Sentence']),
                list(df['Title']),
                list(df['SearchString']),
                list(df['Positive']),
                list(df['Negative']),
                list(df['Neutral']),
                list(df['Compound']),
                list(df['CreatedAT']),
            ):
                sents = {
                    "Sentence": str(sentence),
                    "Title": str(title),
                    "SearchString": str(search),
                    "Positive": str(pos_),
                    "Negative": str(neg_),
                    "Neutral": str(neu_),
                    "Compound": str(com_),
                    "CreatedAT": str(createdAT)
                }
                cleanList.append(sents)
            return cleanList

        except Exception as e:
            print(e)
            self.script_executed = False
            return self.script_executed


class FeaturesClassification:
    def __init__(self, sentence):
        self.sentence = sentence
        self.nlp = spacy.load("en_core_web_sm")
    def PartsOfSpeech(self):
        try:            
            data = []
            for token in self.nlp(self.sentence):
                data.append({   
                    "Text_": str(token.text),
                    "Lemma_": str(token.lemma_),
                    "Pos_": str(token.pos_),
                    "Tag_": str(token.tag_),
                    "Dep_": str(token.dep_),
                    "Shape_": str(token.shape_),
                    "Is_Alpha": str(token.is_alpha),
                    "Is_Stop": str(token.is_stop),            
                })
            return data
        except Exception as e:
            print(f"An error occurred: {e}")
    
    def NounChunk(self):
        try:
            doc = self.nlp(self.sentence)
            dataList = []
            for sent in doc.sents:
                for chunk in sent.noun_chunks:
                    data = {   
                        "Text": str(chunk.text),
                        "RootText": str(chunk.root.text),
                        "RootDep": str(chunk.root.dep_),
                        "RootHead": str(chunk.root.head.text),
                    }
                    dataList.append(data)
            return dataList
        except Exception as e:
            print(f"An error occurred: {e}")
    
    def EntityRecognition(self):
        try:
            doc = self.nlp(self.sentence)
            dataList = []
            for sent in doc.sents:
                for ent in sent.ents:
                    data = {            
                        "Text": str(ent.text),
                        "StartChat": str(ent.start_char),
                        "EndChar": str(ent.end_char),
                        "Label": str(ent.label_),                   
                    }
                    dataList.append(data)
            return dataList
        except Exception as e:
            print(f"An error occurred: {e}")
    
    def Sentiments(self):
        self.script_executed = True
        try:
            sent = self.nlp(self.sentence)
            sentiment = SentimentIntensityAnalyzer()  
            dataList = [] 
            dictionary = sentiment.polarity_scores(str(sent))
            positive = str(round(float(dictionary['pos']), 4))
            negative = str(round(float(dictionary['neg']), 4))
            neutral = str(round(float(dictionary['neu']), 4))
            compound = str(round(float(dictionary['compound']), 4))
            data = {   
                "Sentence": str(self.sentence),
                "Positive": str(positive),
                "Negative": str(negative),
                "Neutral": str(neutral),
                "Compound": str(compound),              
            }
            dataList.append(data)
            return dataList
        except Exception as e:
            print(e)
            self.script_executed = False
            return self.script_executed
        
    def question_answering(self, context):
        question = str(context).split('split_text')[0] + '?'
        sentence = str(context).split('split_text')[1]
        information_extraction_pipeline = pipeline("question-answering")
        extract_info = information_extraction_pipeline(question=f"{question}", context=sentence)

        data = {
            "Answer": str(extract_info["answer"]),
            "End": str(extract_info["end"]),
            "Score": str(extract_info["score"]),
            "Start": str(extract_info["start"])
        }
        print(json.dumps(data, indent=4))
        return data 