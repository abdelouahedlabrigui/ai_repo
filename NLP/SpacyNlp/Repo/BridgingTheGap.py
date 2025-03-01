import datetime
import json
from utils import *
from transformers import pipeline, set_seed
import pandas as pd
set_seed(42)

class TextClassification:
    def __init__(self, title: str, text: str):
        self.text = text
        self.title = title
    def text_classification(self, sector):
        try:
            if sector == "medical":
                if (not self.text.strip()) or (not self.title.strip()):
                    raise ValueError("Input text is empty!")
                classifier = pipeline("text-classification")
                outputs = classifier(self.text)
                df = pd.DataFrame(outputs)
                data = []
                for label, score in zip(list(df["label"]), list(df["score"])):
                    data.append({
                        "Title": self.title, "Text": self.text, "Label": label, "Score": score, 
                        "CreatedAT": str(datetime.datetime.now())
                    })
                return data
        except Exception as e:
            raise ValueError(f"{e}")

class NerTagger:
    def __init__(self, title: str, text: str):
        self.text = text
        self.title = title

    def named_entity_recognition(self, sector):
        try:
            if sector == "medical":
                if (not self.text.strip()) or (not self.title.strip()):
                    raise ValueError("Input text is empty!")

                ner_tagger = pipeline("ner", model="dslim/distilbert-NER", aggregation_strategy="simple")
                outputs = ner_tagger(self.text)

                # print("NER Output:", outputs)  # Debugging print

                if not outputs:
                    print("No named entities found!")
                    return []

                df = pd.DataFrame(outputs)
                print(df)

                data = [
                    {
                        "Title": self.title,
                        "EntityGroup": row["entity_group"],
                        "Score": row["score"],
                        "Word": row["word"],
                        "Start": row["start"],
                        "End": row["end"],
                        "CreatedAT": str(datetime.datetime.now())
                    }
                    for _, row in df.iterrows()
                ]

                return data
        except Exception as e:
            raise ValueError(f"Error: {e}")
  
# ner_tagger = pipeline("ner", aggregation_strategy="simple")

class QuestionAnswering:
    def __init__(self, title: str, text: str, question: str):
        self.text = text
        self.question = question
        self.title = title
    def question_answering(self, sector):
        try:
            if sector == "medical":
                if (not self.text.strip()) or (not self.title.strip()) or (not self.question.strip()):
                    raise ValueError("Input text is empty!")
                reader = pipeline("question-answering", model="Falconsai/question_answering_v2")
                outputs = reader(question=self.question, context=self.text)
                df = pd.DataFrame([outputs])
                data = []
                for score, start, end, answer in zip(list(df["score"]), list(df["start"]), list(df["end"]), list(df["answer"])):
                    data.append({"Title": self.title, "Question": self.question, 'Score': score, 'Start': start, 'End': end, 'Answer': answer, 
                    "CreatedAT": str(datetime.datetime.now())})
                return data
        except Exception as e:
            raise ValueError(f"{e}")


class Summarization:
    def __init__(self, title: str, text: str):
        self.text = text
        self.title = title
    def summary_generator(self, sector):
        try:
            if sector == "medical":
                if (not self.text.strip()) or (not self.title.strip()):
                    raise ValueError("Input text is empty!")
                summarizer = pipeline("summarization", model="Falconsai/text_summarization")
                outputs = summarizer(self.text, max_length=300, clean_up_tokenization_spaces=True)
                summary_text = outputs[0]["summary_text"]
                data = []
                data.append({
                    "Title": self.title, "Text": self.text, "SummaryText": str(summary_text), 
                    "CreatedAT": str(datetime.datetime.now())
                })
                return data
        except Exception as e:
            raise ValueError(f"{e}")

class Translation:
    def __init__(self, title: str, text: str):
        self.text = text
        self.title = title
    def translation_generator(self, sector):
        try:
            if sector == "medical":
                if (not self.text.strip()) or (not self.title.strip()):
                    raise ValueError("Input text is empty!")
                translator = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-es")
                outputs = translator(self.text, clean_up_tokenization_spaces=True, min_length=150)
                data = []
                translation_text = outputs[0]['translation_text']
                data.append({
                    "Title": self.title, "Text": self.text, "TextLanguage": "English", "TranslationText": translation_text, "TranslationLanguage": "Spanish", 
                    "CreatedAT": str(datetime.datetime.now())})
                return data
        except Exception as e:
            raise ValueError(f"{e}")

class TextGeneration:
    def __init__(self, title: str, text: str, actor: str, response: str):
        self.text = text
        self.actor = actor
        self.response = response
        self.title = title
    def text_generation(self, sector):
        try:
            if sector == "medical":
                if (not self.text.strip()) or (not self.title.strip()) or (not self.actor.strip()) or (not self.response.strip()):
                    raise ValueError("Input text is empty!")
                generator = pipeline('text-generation')
                prompt = self.text + f"\n\n based on perspective of: {self.actor}:\n" + self.response
                outputs = generator(prompt, max_length=300)
                generated_text = outputs[0]["generated_text"]
                data = []
                data.append({
                    "Title": self.title, "Text": self.text, "Actor": self.actor, "Response": self.response, "GeneratedText": generated_text, 
                    "CreatedAT": str(datetime.datetime.now())
                })
                return data
        except Exception as e:
            raise ValueError(f"{e}")