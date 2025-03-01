import datetime
import json
import os
import numpy as np
import spacy
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import wikipediaapi

class FeatureExtraction:
    def __init__(self, text_file):
        self.nlp = spacy.load("en_core_web_sm")
        self.text_file = text_file
        self.doc = self.nlp(str(open(self.text_file, encoding='utf-8').read()))
        self.df = self.extract_entities()
        self.file_name = str(os.path.basename(self.text_file))        

    def extract_entities(self):
        entities = []
        for sent in self.doc.sents:
            for ent in sent.ents:
                data = {          
                    "Text": str(ent.text),
                    "StartChat": str(ent.start_char),
                    "EndChar": str(ent.end_char),
                    "Label": str(ent.label_)                  
                }
                entities.append(data)

        df = pd.DataFrame(entities)
        return df
    
    def filter_person(self):
        person_df = self.df.loc[self.df['Label'] == "PERSON"]
        person_df = person_df.loc[person_df["Text"].str.contains(" ")]
        person_df = person_df[["Text", "Label"]].drop_duplicates()
        return person_df
    
    def filter_non_gpe(self):
        non_gpe_df = self.df.loc[self.df['Label'] == "LOC"]
        non_gpe_df = non_gpe_df.loc[non_gpe_df["Text"].str.contains(" ")]
        non_gpe_df = non_gpe_df[["Text", "Label"]].drop_duplicates()
        return non_gpe_df
    
    def filter_nationalities(self):
        nationalities_df = self.df.loc[self.df['Label'] == "NORP"]
        nationalities_df = nationalities_df.loc[nationalities_df["Text"].str.contains(" ")]
        nationalities_df = nationalities_df[["Text", "Label"]].drop_duplicates()
        return nationalities_df
    
    def filter_buildings(self):
        buildings_df = self.df.loc[self.df['Label'] == "FAC"]
        buildings_df = buildings_df.loc[buildings_df["Text"].str.contains(" ")]
        buildings_df = buildings_df[["Text", "Label"]].drop_duplicates()
        return buildings_df
    
    def filter_companies(self):
        companies_df = self.df.loc[self.df['Label'] == "ORG"]
        companies_df = companies_df.loc[companies_df["Text"].str.contains(" ")]
        companies_df = companies_df[["Text", "Label"]].drop_duplicates()
        return companies_df
    
    def filter_countries(self):
        countries_df = self.df.loc[self.df['Label'] == "GPE"]
        countries_df = countries_df.loc[countries_df["Text"].str.contains(" ")]
        countries_df = countries_df[["Text", "Label"]].drop_duplicates()
        return countries_df
    
    def select_entity(self, label):
        if label == "person":
            return self.filter_person()
        elif label == "non_gpe":
            return self.filter_non_gpe()
        elif label == "nationalities":
            return self.filter_nationalities()
        elif label == "buildings":
            return self.filter_buildings()
        elif label == "companies":
            return self.filter_companies()
        elif label == "countries":
            return self.filter_countries()

    
    def extract_label_context(self, doc_text, label=None, search_string=None):
        results = []
        matches = list(re.finditer(re.escape(search_string), doc_text, re.IGNORECASE))

        for match in matches:
            start = max(0, match.start() - 300)  
            end = min(len(doc_text), match.end() + 300)

            # Adjust to nearest whitespace to avoid cutting words
            while start > 0 and doc_text[start].isalnum():
                start -= 1
            while end < len(doc_text) and doc_text[end - 1].isalnum():
                end += 1

            context = doc_text[start:end].strip()

            if label == "person":
                results.append({'Person': search_string, 'Context': context, "FileName": self.file_name, "CreatedAT": str(datetime.datetime.now())})
            elif label == "non_gpe":
                results.append({'NonGpe': search_string, 'Context': context, "FileName": self.file_name, "CreatedAT": str(datetime.datetime.now())})
            elif label == "nationalities":
                results.append({'Nationalities': search_string, 'Context': context, "FileName": self.file_name, "CreatedAT": str(datetime.datetime.now())})
            elif label == "buildings":
                results.append({'Buildings': search_string, 'Context': context, "FileName": self.file_name, "CreatedAT": str(datetime.datetime.now())})
            elif label == "companies":
                results.append({'Companies': search_string, 'Context': context, "FileName": self.file_name, "CreatedAT": str(datetime.datetime.now())})
            elif label == "countries":
                results.append({'Countries': search_string, 'Context': context, "FileName": self.file_name, "CreatedAT": str(datetime.datetime.now())})

        df = pd.DataFrame(results) #creates dataframe from results list.
        return df, results  

    def extract_chunks(self, df):
        """
        Extracts noun chunks from the 'Context' column of a DataFrame.

        Args:
            self: Instance of the class containing this method, which must have a self.nlp spacy object.
            df (pd.DataFrame): DataFrame with 'Context' column.

        Returns:
            pd.DataFrame: DataFrame with added 'selected_chunks' column.
        """

        chunks_list = []

        for context in df['Context']:
            doc = self.nlp(context)
            noun_chunks = [chunk.text for chunk in doc.noun_chunks]
            chunks_list.append(", ".join(noun_chunks))

        df['selected_chunks'] = chunks_list
        return df
    
    def match_label_context(self, doc_text, label=None, entity_search_string=None, search_string=None):        
        if label == "person":
            # df = self.filter_person()
            df, results = self.extract_label_context(doc_text, "person", entity_search_string)
            # df = df.loc[df["Context"].str.contains(entity_search_string)]
            df = self.extract_chunks(df=df)            
            data = []
            for person, context, selected_chunks in zip(
                list(df['Person']),
                list(df['Context']),
                list(df['selected_chunks'])):
                data.append({"Person": person, "Context": context, "SearchTerm": "empty", "SelectedChunks": selected_chunks,
                    "FileName": self.file_name, "CreatedAT": str(datetime.datetime.now())})
            return data
        elif label == "non_gpe":
            # df = self.filter_non_gpe()
            df, results = self.extract_label_context(doc_text, "non_gpe", entity_search_string)
            # df = df.loc[df["Context"].str.contains(entity_search_string)]
            df = self.extract_chunks(df=df)
            data = []
            for non_gpe, context, selected_chunks in zip(
                list(df['NonGpe']),
                list(df['Context']),
                list(df['selected_chunks'])):
                data.append({"NonGpe": non_gpe, "Context": context, "SearchTerm": "empty", "SelectedChunks": selected_chunks,
                    "FileName": self.file_name, "CreatedAT": str(datetime.datetime.now())})
            return data
        elif label == "nationalities":
            # df = self.filter_nationalities()
            df, results = self.extract_label_context(doc_text, "nationalities", entity_search_string)
            # df = df.loc[df["Context"].str.contains(entity_search_string)]
            df = self.extract_chunks(df=df)
            data = []
            for nationalities, context, selected_chunks in zip(
                list(df['Nationalities']),
                list(df['Context']),
                list(df['selected_chunks'])):
                data.append({"Nationalities": nationalities, "Context": context, "SearchTerm": "empty", "SelectedChunks": selected_chunks,
                    "FileName": self.file_name, "CreatedAT": str(datetime.datetime.now())})
            return data
        elif label == "buildings":
            # df = self.filter_buildings()
            df, results = self.extract_label_context(doc_text, "buildings", entity_search_string)
            # df = df.loc[df["Context"].str.contains(entity_search_string)]
            df = self.extract_chunks(df=df)
            data = []
            for buildings, context, selected_chunks in zip(
                list(df['Buildings']),
                list(df['Context']),
                list(df['selected_chunks'])):
                data.append({"Buildings": buildings, "Context": context, "SearchTerm": "empty", "SelectedChunks": selected_chunks,
                    "FileName": self.file_name, "CreatedAT": str(datetime.datetime.now())})
            return data
        elif label == "companies":
            # df = self.filter_companies()
            df, results = self.extract_label_context(doc_text, "companies", entity_search_string)
            # df = df.loc[df["Context"].str.contains(entity_search_string)]
            df = self.extract_chunks(df=df)
            data = []
            for companies, context, selected_chunks in zip(
                list(df['Companies']),
                list(df['Context']),
                list(df['selected_chunks'])):
                data.append({"Companies": companies, "Context": context, "SearchTerm": "empty", "SelectedChunks": selected_chunks,
                    "FileName": self.file_name, "CreatedAT": str(datetime.datetime.now())})
            return data
        elif label == "countries":
            # df = self.filter_countries()
            df, results = self.extract_label_context(doc_text, "countries", entity_search_string)
            # df = df.loc[df["Context"].str.contains(entity_search_string)]
            df = self.extract_chunks(df=df)
            data = []
            for countries, context, selected_chunks in zip(
                list(df['Countries']),
                list(df['Context']),
                list(df['selected_chunks'])):
                data.append({"Countries": countries, "Context": context, "SearchTerm": "empty", "SelectedChunks": selected_chunks,
                    "FileName": self.file_name, "CreatedAT": str(datetime.datetime.now())})
            return data
        
class NLPTransformers:
    def __init__(self):
        pass
    def question_answering(self, context):
        information_extraction_pipeline = pipeline("question-answering")
        extract_info = information_extraction_pipeline(question=f"{context}", context=context)
        return extract_info
    def sentiment_results(self, context):
        sentiment_pipeline = pipeline("sentiment-analysis")
        sentiment_result = sentiment_pipeline(context)
        return sentiment_result

    def summarization_results(self, context):
        summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
        summary_result = summarization_pipeline(context, max_length=(len(context) + 30), min_length=30, do_sample=False)
        return summary_result
    
    def information_extraction_results(self, context):
        information_extraction_pipeline = pipeline("question-answering")
        extract_info = information_extraction_pipeline(question=f"What is the main topic in this text: {context}?", context=context)
        return extract_info

    def event_results(self, context):
        event_pipeline = pipeline("question-answering")
        event_result = event_pipeline(question="What events are mentioned?", context=context)
        return event_result
    
    def features_extraction(self, context):
        data = {}
        sentiment_result = self.sentiment_results(context)
        extract_info = self.information_extraction_results(context)
        event_result = self.event_results(context)
        
        data["Context"] = context
        data['Sentiment'] = sentiment_result[0]['label']
        data['SentimentScore'] = str(sentiment_result[0]['score'])
        data['ExtractedAnswer'] = extract_info['answer']
        data['ExtractedScore'] = str(extract_info['score'])
        data['Events'] = event_result['answer']
        data['EventScore'] = str(event_result['score'])
        data["CreatedAT"] = str(datetime.datetime.now())

        return data  # Return the dictionary containing all extracted information
    
class WikipediaDocuments:
    def __init__(self, search_string):
        self.search_string = search_string
        self.user_agent = "Abdelouahedlabrigui"

    def document_search(self):
        wiki_wiki = wikipediaapi.Wikipedia(user_agent=self.user_agent, language='en')
        page = wiki_wiki.page(self.search_string)

        if not page.exists():
            return {"error": "Person not found on Wikipedia"}

        data = {
            "SearchString": self.search_string,
            "Title": page.title,
            "Summary": page.summary + "..." if len(page.summary) > 500 else page.summary,
            "Text": page.text,
            "CreatedAT": str(datetime.datetime.now())
        }
        return data

if __name__ == "__main__":
    file = r"C:\Users\dell\Entrepreneurship\Engineering\ai_repo\Data\History\text.txt"
    features = FeatureExtraction(text_file=file)
    df = features.select_entity(label="companies")
    doc_text = str(features.doc)
    data = features.match_label_context(doc_text=doc_text, label="companies", entity_search_string="the British Commonwealth", search_string="independence")
    print(json.dumps(data, indent=4))
