import datetime
import pandas as pd
import spacy
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



class SpacyNlpSentMessage:
    def __init__(self, prompt):
        self.prompt = prompt
        self.nlp = spacy.load("en_core_web_sm")
    def EntityRecognition(self):
        try:
            doc = self.nlp(self.prompt)
            dataList = []
            for sent in doc.sents:
                for ent in sent.ents:
                    data = {          
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
            doc = self.nlp(self.prompt)
            dataList = []
            for sent in doc.sents:
                for chunk in sent.noun_chunks:
                    if chunk.root.pos_ != "AUX" and chunk.root.pos_ != "NUM" and chunk.root.pos_ != "PRON" and chunk.root.pos_ != "PROPN" and chunk.root.pos_ != "PUNCT" and chunk.root.pos_ != "SYM":
                        data = {   
                            "Text": str(chunk.text),                        
                            "RootText": str(chunk.root.text),
                            "RootDep": str(chunk.root.dep_),
                            "RootHead": str(chunk.root.head.text),
                            "CreatedAT": str(datetime.datetime.now().isoformat())
                        }
                        dataList.append(data)
            # print(json.dumps(dataList, indent=4))
            return dataList
        except Exception as e:
            print(f"An error occurred: {e}")
    def EntitiesFilter(self):
        try:
            doc = self.nlp(self.prompt)
            dataList = []
            for sent in doc.sents:
                for ent in sent.ents:
                    data = {   
                        "Text": str(ent.text),
                        "StartChar": str(ent.start_char),
                        "EndChar": str(ent.end_char),
                        "Label": str(ent.label_), 
                        "CreatedAT": str(datetime.datetime.now().isoformat())
                    }
                    dataList.append(data)   
            return dataList
        except Exception as e:
            print(f"An error occurred: {e}")
    def Sentiments(self):
        self.script_executed = True
        try:
            doc = self.nlp(self.prompt)
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
        
class SpanishSpacyNlpSentMessage:
    def __init__(self, prompt):
        self.prompt = prompt
        self.nlp = spacy.load("es_core_news_sm")
        
    def EntityRecognition(self):
        try:
            doc = self.nlp(self.prompt)
            dataList = []
            for sent in doc.sents:
                for ent in sent.ents:
                    data = {          
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
            doc = self.nlp(self.prompt)
            dataList = []
            for sent in doc.sents:
                for chunk in sent.noun_chunks:
                    if chunk.root.pos_ != "AUX" and chunk.root.pos_ != "NUM" and chunk.root.pos_ != "PRON" and chunk.root.pos_ != "PROPN" and chunk.root.pos_ != "PUNCT" and chunk.root.pos_ != "SYM":
                        data = {   
                            "Text": str(chunk.text),                        
                            "RootText": str(chunk.root.text),
                            "RootDep": str(chunk.root.dep_),
                            "RootHead": str(chunk.root.head.text),
                            "CreatedAT": str(datetime.datetime.now().isoformat())
                        }
                        dataList.append(data)
            # print(json.dumps(dataList, indent=4))
            return dataList
        except Exception as e:
            print(f"An error occurred: {e}")
    def Sentiments(self):
        self.script_executed = True
        try:
            doc = self.nlp(self.prompt)
            sentiment = SentimentIntensityAnalyzer()  
            dataList = [] 
            for sent in str(doc).split('.'):   
                dictionary = sentiment.polarity_scores(str(sent))
                positive = str(round(float(dictionary['pos']), 4))
                negative = str(round(float(dictionary['neg']), 4))
                neutral = str(round(float(dictionary['neu']), 4))
                compound = str(round(float(dictionary['compound']), 4))
                data = {   
                    "Sentence": str(sent),
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