import datetime
import os
from flask import Flask, json, jsonify, request
import pandas as pd
from Repo.SpacyNlp import SpacyNlp, TextGeneration, FeaturesClassification
from Repo.SentMessage import SpacyNlpSentMessage, SpanishSpacyNlpSentMessage
from Repo.Preprocessing.FeatureExtraction import FeatureExtraction, NLPTransformers, WikipediaDocuments
# from Repo.SpacyNlp import Translation
from Repo.BridgingTheGap import TextClassification
from Repo.BridgingTheGap import NerTagger
from Repo.BridgingTheGap import QuestionAnswering
from Repo.BridgingTheGap import Summarization
from Repo.BridgingTheGap import Translation
from Repo.BridgingTheGap import TextGeneration
from flask_cors import CORS
from urllib.parse import unquote

app = Flask(__name__)

CORS(app)

@app.route('/llms/text_classification', methods=['GET'])
def text_classification():
    try:
        title = unquote(request.args.get("title"))
        text = unquote(request.args.get('text'))
        sector = unquote(request.args.get("sector")) 
        if (not title) or (not text) or (not sector):
            return jsonify({"Message": "One or more parameters not valid"}), 404    
        init = TextClassification(title=title, text=text)
        data = init.text_classification(sector)

        # Convert to DataFrame
        df = pd.DataFrame(data)
        print(df)

        # Check for empty DataFrame
        if df.empty:
            return jsonify({"Message": "No named entities found"}), 404

        # Check for null values
        if df.isnull().values.any():
            return jsonify({"Message": "Dataset contains missing values"}), 404  

        return jsonify(data), 200
    
    except ValueError as ve:
        return jsonify({"Message": str(ve)}), 400  # Client error
    except Exception as e:
        print(f"Server Error: {str(e)}")  # Log for debugging
        return jsonify({"Message": "Internal server error"}), 500  # Server error
    
@app.route('/llms/ner_tagger', methods=['GET'])
def ner_tagger():
    try:
        # Decode URL parameters
        title = unquote(request.args.get("title", ""))
        text = unquote(request.args.get("text", ""))
        sector = unquote(request.args.get("sector", ""))

        # Validate input parameters
        if not title or not text or not sector:
            return jsonify({"Message": "One or more parameters not valid"}), 400  

        # Initialize NER model
        init = NerTagger(title=title, text=text)
        data = init.named_entity_recognition(sector)  

        # Convert to DataFrame
        df = pd.DataFrame(data)
        print(df)

        # Check for empty DataFrame
        if df.empty:
            return jsonify({"Message": "No named entities found"}), 404

        # Check for null values
        if df.isnull().values.any():
            return jsonify({"Message": "Dataset contains missing values"}), 404  

        return jsonify(data), 200
    
    except ValueError as ve:
        return jsonify({"Message": str(ve)}), 400  # Client error
    except Exception as e:
        print(f"Server Error: {str(e)}")  # Log for debugging
        return jsonify({"Message": "Internal server error"}), 500  # Server error

    
@app.route('/llms/question_answering', methods=['GET'])
def question_answering():
    try:
        title = unquote(request.args.get("title"))
        text = unquote(request.args.get('text'))
        question = unquote(request.args.get("question"))
        sector = unquote(request.args.get("sector"))
        if (not title) or (not text) or (not question) or (not sector):
            return jsonify({"Message": "One or more parameters not valid"}), 404 
        init = QuestionAnswering(title=title, text=text, question=question)
        data = init.question_answering(sector)

        # Convert to DataFrame
        df = pd.DataFrame(data)
        print(df)

        # Check for empty DataFrame
        if df.empty:
            return jsonify({"Message": "No named entities found"}), 404

        # Check for null values
        if df.isnull().values.any():
            return jsonify({"Message": "Dataset contains missing values"}), 404  

        return jsonify(data), 200
    
    except ValueError as ve:
        return jsonify({"Message": str(ve)}), 400  # Client error
    except Exception as e:
        print(f"Server Error: {str(e)}")  # Log for debugging
        return jsonify({"Message": "Internal server error"}), 500  # Server error
    
@app.route('/llms/summarization', methods=['GET'])
def summarization():
    try:
        title = unquote(request.args.get("title"))
        text = unquote(request.args.get('text'))
        sector = unquote(request.args.get("sector"))
        if (not title) or (not text) or (not sector):
            return jsonify({"Message": "One or more parameters not valid"}), 404 
        init = Summarization(title=title, text=text)
        data = init.summary_generator(sector)

        # Convert to DataFrame
        df = pd.DataFrame(data)
        print(df)

        # Check for empty DataFrame
        if df.empty:
            return jsonify({"Message": "No named entities found"}), 404

        # Check for null values
        if df.isnull().values.any():
            return jsonify({"Message": "Dataset contains missing values"}), 404  

        return jsonify(data), 200
    
    except ValueError as ve:
        return jsonify({"Message": str(ve)}), 400  # Client error
    except Exception as e:
        print(f"Server Error: {str(e)}")  # Log for debugging
        return jsonify({"Message": "Internal server error"}), 500  # Server error
    
@app.route('/llms/translation', methods=['GET'])
def translation():
    try:
        title = unquote(request.args.get("title"))
        text = unquote(request.args.get('text'))
        sector = unquote(request.args.get("sector"))
        if (not title) or (not text) or (not sector):
            return jsonify({"Message": "One or more parameters not valid"}), 404 
        init = Translation(title=title, text=text)
        data = init.translation_generator(sector)

        # Convert to DataFrame
        df = pd.DataFrame(data)
        print(df)

        # Check for empty DataFrame
        if df.empty:
            return jsonify({"Message": "No named entities found"}), 404

        # Check for null values
        if df.isnull().values.any():
            return jsonify({"Message": "Dataset contains missing values"}), 404  

        return jsonify(data), 200
    
    except ValueError as ve:
        return jsonify({"Message": str(ve)}), 400  # Client error
    except Exception as e:
        print(f"Server Error: {str(e)}")  # Log for debugging
        return jsonify({"Message": "Internal server error"}), 500  # Server error
    
@app.route('/llms/text_generation', methods=['GET'])
def text_generation():
    try:
        title = unquote(request.args.get("title"))
        text = unquote(request.args.get('text'))
        actor = unquote(request.args.get("actor"))
        response = unquote(request.args.get("response"))
        sector = unquote(request.args.get("sector"))
        if (not title) or (not text) or (not actor) or (not response) or (not sector):
            return jsonify({"Message": "One or more parameters not valid"}), 404 
        
        init = TextGeneration(title=title, text=text, actor=actor, response=response)
        data = init.text_generation(sector)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        print(df)

        # Check for empty DataFrame
        if df.empty:
            return jsonify({"Message": "No named entities found"}), 404

        # Check for null values
        if df.isnull().values.any():
            return jsonify({"Message": "Dataset contains missing values"}), 404  

        return jsonify(data), 200
    
    except ValueError as ve:
        return jsonify({"Message": str(ve)}), 400  # Client error
    except Exception as e:
        print(f"Server Error: {str(e)}")  # Log for debugging
        return jsonify({"Message": "Internal server error"}), 500  # Server error

@app.route("/nlp/wikipedia_document_search")
def wikipedia_document_search():
    try:
        search_string = unquote(request.args.get('search_string'))
        wiki = WikipediaDocuments(search_string=search_string)
        data = wiki.document_search()
        return jsonify([data]), 200 
    except Exception as e:
        print({ "Message": f"Error: {str(e)}" })
        return jsonify([{"Message": f"Error message, status code: 500"}]), 500

@app.route("/nlp/extract_entities")
def extract_entities():
    try:
        text_file = unquote(request.args.get('text_file'))
        label = unquote(request.args.get('label'))
        if ".txt" not in text_file or os.path.exists(text_file) == False:
            return jsonify([{"Message": f"Error checking text file."}])
        # if (label != "person") or (label != "non_gpe") or (label != "nationalities") or (label != "buildings") or (label != "companies") or (label != "countries"):
        #     return jsonify([{"Message": f"None of these labels found: person, non_gpe, nationalities, buildings, companies, countries"}])
        
        features = FeatureExtraction(text_file=text_file)
        df = features.select_entity(label=label)
        results = []
        file_name = features.file_name
        for text, label in zip(list(df["Text"]), list(df["Label"])):
            results.append({"Text": text, "Label": label, "FileName": file_name, "CreatedAT": str(datetime.datetime.now())})
        return jsonify(results), 200
    except Exception as e:
        print({ "Message": f"Error: {str(e)}" })
        return jsonify([{"Message": f"Error message, status code: 500"}]), 500


@app.route("/nlp/extract_label_context")
def extract_label_context():
    try:
        text_file = unquote(request.args.get('text_file'))
        label = unquote(request.args.get('label'))
        if ".txt" not in text_file or os.path.exists(text_file) == False:
            return jsonify([{"Message": f"Error checking text file."}])
        # if (label != "person") or (label != "non_gpe") or (label != "nationalities") or (label != "buildings") or (label != "companies") or (label != "countries"):
        #     return jsonify([{"Message": f"None of these labels found: person, non_gpe, nationalities, buildings, companies, countries"}])
        
        features = FeatureExtraction(text_file=text_file)
        df = features.select_entity(label=label)
        doc_text = str(features.doc)
        df, results = features.extract_label_context(df=df, doc_text=doc_text, label=label)
        return jsonify(results), 200
    except Exception as e:
        print({ "Message": f"Error: {str(e)}" })
        return jsonify([{"Message": f"Error message, status code: 500"}]), 500
    
@app.route("/nlp/match_label_context")
def match_label_context():
    try:
        text_file = unquote(request.args.get('text_file'))
        label = unquote(request.args.get('label'))
        entity_search_string = unquote(request.args.get('entity_search_string'))
        search_string = unquote(request.args.get('search_string'))
        
        if ".txt" not in text_file or os.path.exists(text_file) == False:
            return jsonify([{"Message": f"Error checking text file."}])
        
        # if (label != "person") or (label != "non_gpe") or (label != "nationalities") or (label != "buildings") or (label != "companies") or (label != "countries"):
        #     return jsonify([{"Message": f"None of these labels found: person, non_gpe, nationalities, buildings, companies, countries"}])
        
        if len(entity_search_string) <= 8 or search_string == "":
            return jsonify([{"Message": f"Error checking inputs: entity_search_string, search_string."}])
        
        features = FeatureExtraction(text_file=text_file)
        doc_text = str(features.doc)
        data = features.match_label_context(doc_text=doc_text, label=label, entity_search_string=entity_search_string, search_string=search_string)
        return jsonify(data), 200
    except Exception as e:
        print({ "Message": f"Error: {str(e)}" })
        return jsonify([{"Message": f"Error message, status code: 500"}]), 500
    
@app.route("/nlp/features_extraction")
def features_extraction():
    try:
        context = unquote(request.args.get('context'))
        transformers = NLPTransformers()
        data = transformers.features_extraction(context=context)
        return jsonify([data]), 200
    except Exception as e:
        print({ "Message": f"Error: {str(e)}" })
        return jsonify([{"Message": f"Error message, status code: 500"}]), 500

@app.route("/nlp/generate_sentiments")
def generate_sentiments():
    try:
        title = unquote(request.args.get('title'))
        searchString = unquote(request.args.get('searchString'))
        text = unquote(request.args.get('text'))
        
        spacyNlp = SpacyNlp(title, searchString, text)
        data = spacyNlp.Sentiments()
        print(json.dumps(data, indent=4))
        return jsonify(data), 200        
    except Exception as e:
        print({ "Message": f"Error: {str(e)}" })
        return jsonify([{"Message": f"Error message, status code: 500"}]), 500
    
@app.route("/nlp/querying_sentiments")
def querying_sentiments():
    try:
        title = request.args.get('title')
        searchString = request.args.get('searchString')
        text = request.args.get('text')
        state = request.args.get('state')

        spacyNlp = SpacyNlp(title, searchString, text)
        data = spacyNlp.SentimentsQuerying(state)
        print(json.dumps(data, indent=4))
        return jsonify(data), 200        
    except Exception as e:
        print({ "Message": f"Error: {str(e)}" })
        return jsonify([{"Message": f"Error message, status code: 500"}]), 500

@app.route("/nlp/generate_prompt_noun_chunks")
def generate_prompt_noun_chunks():
    try:
        prompt = request.args.get('prompt')
        prompt = prompt.replace("This is a prompt:", "")
        spacyNlp = SpacyNlpSentMessage(prompt)
        data = spacyNlp.NounChunk()

        return jsonify(data), 200       
    except Exception as e:
        print({ "Message": f"Error: {str(e)}" })
        return jsonify([{"Message": f"Error message, status code: 500"}]), 500
    
@app.route("/nlp/generate_pos_tags")
def generate_pos_tags():
    try:
        sentence = unquote(request.args.get('sentence'))
        query = FeaturesClassification(sentence=sentence)
        data = query.PartsOfSpeech()

        return jsonify(data), 200 
    except Exception as e:
        print({ "Message": f"Error: {str(e)}" })
        return jsonify([{"Message": f"Error message, status code: 500"}]), 500
    
@app.route("/nlp/generate_noun_chunks_tags")
def generate_noun_chunks_tags():
    try:
        sentence = unquote(request.args.get('sentence'))
        query = FeaturesClassification(sentence=sentence)
        data = query.NounChunk()

        return jsonify(data), 200 
    except Exception as e:
        print({ "Message": f"Error: {str(e)}" })
        return jsonify([{"Message": f"Error message, status code: 500"}]), 500
    
@app.route("/nlp/generate_entities_tags")
def generate_entities_tags():
    try:
        sentence = unquote(request.args.get('sentence'))
        query = FeaturesClassification(sentence=sentence)
        data = query.EntityRecognition()

        return jsonify(data), 200 
    except Exception as e:
        print({ "Message": f"Error: {str(e)}" })
        return jsonify([{"Message": f"Error message, status code: 500"}]), 500
    
@app.route("/nlp/generate_sentiments_score")
def generate_sentiments_score():
    try:
        sentence = unquote(request.args.get('sentence'))
        query = FeaturesClassification(sentence=sentence)
        data = query.Sentiments()

        return jsonify(data), 200 
    except Exception as e:
        print({ "Message": f"Error: {str(e)}" })
        return jsonify([{"Message": f"Error message, status code: 500"}]), 500
    
# question_answering
@app.route("/nlp/generate_question_answering")
def generate_question_answering():
    try:
        context = unquote(request.args.get('context'))
        query = FeaturesClassification(sentence=context)
        data = query.question_answering(context)

        return jsonify(data), 200 
    except Exception as e:
        print({ "Message": f"Error: {str(e)}" })
        return jsonify([{"Message": f"Error message, status code: 500"}]), 500
    

@app.route("/nlp/generate_prompt_entities_by_filter")
def generate_prompt_entities_by_filter():
    try:
        prompt = unquote(request.args.get('prompt'))
        prompt = prompt.replace("This is a prompt:", "")
        spacyNlp = SpacyNlpSentMessage(prompt)
        data = spacyNlp.EntitiesFilter()
        return jsonify(data), 200       
    except Exception as e:
        print({ "Message": f"Error: {str(e)}" })
        return jsonify([{"Message": f"Error message, status code: 500"}]), 500

@app.route("/nlp/generate_prompt_sentiments")
def generate_prompt_sentiments():
    try:
        prompt = unquote(request.args.get('prompt'))
        prompt = prompt.replace("This is a prompt:", "")
        spacyNlp = SpacyNlpSentMessage(prompt)
        data = spacyNlp.Sentiments()

        return jsonify(data), 200       
    except Exception as e:
        print({ "Message": f"Error: {str(e)}" })
        return jsonify([{"Message": f"Error message, status code: 500"}]), 500
    
# @app.route('/llms/text_generation', methods=['GET'])
# def text_generation():
#     try:
#         text = unquote(request.args.get('text'))
#         actor = unquote(request.args.get("actor"))
#         response = unquote(request.args.get("response"))
#         if (not text) or (not actor) or (not response):
#             return jsonify({"Message": "One or more parameters not valid"}), 404 
        
#         init = Translation(text=text, actor=actor, response=response)
#         data = init.translation_generator()
        
#         return jsonify(data), 200    
#     except ValueError as ve:
#         return jsonify({"Message": str(ve)}), 400  # Client error
#     except Exception as e:
#         print(f"Server Error: {str(e)}")  # Log for debugging
#         return jsonify({"Message": "Internal server error"}), 500  # Server error
    
@app.route('/llms/text_generation_without_translation', methods=['GET'])
def text_generation_without_translation():
    try:
        text = unquote(request.args.get('text'))
        actor = unquote(request.args.get("actor"))
        response = unquote(request.args.get("response"))
        # if (not text) or (not actor) or (not response):
        #     return jsonify({"Message": "One or more parameters not valid"}), 404 
        
        text_generation = TextGeneration(text=text, actor=actor, response=response).text_generation()
        data = {
            "Text": text_generation
        }
        return jsonify([data]), 200    
    except ValueError as ve:
        return jsonify({"Message": str(ve)}), 400  # Client error
    except Exception as e:
        print(f"Server Error: {str(e)}")  # Log for debugging
        return jsonify({"Message": "Internal server error"}), 500  # Server error
    
@app.route("/nlp/generate_raw_entities")
def generate_raw_entities():
    try:
        english_text = unquote(request.args.get('english_text'))
        spanish_text = unquote(request.args.get('spanish_text'))

        if english_text == "empty" or spanish_text == "empty":
            return jsonify([{"Message": f"Param empty."}]), 500

        spacyNlp = SpacyNlpSentMessage(english_text)
        english = spacyNlp.EntityRecognition()
        
        spanishSpacyNlp = SpanishSpacyNlpSentMessage(spanish_text)
        spanish = spanishSpacyNlp.EntityRecognition()

        return jsonify([{"English": english, "Spanish": spanish}]), 200       
    except Exception as e:
        print({ "Message": f"Error: {str(e)}" })
        return jsonify({"Message": f"Error message, status code: 500"}), 500
    
@app.route("/nlp/generate_raw_noun_chunks")
def generate_raw_noun_chunks():
    try:
        english_text = unquote(request.args.get('english_text'))
        spanish_text = unquote(request.args.get('spanish_text'))

        if english_text == "empty" or spanish_text == "empty":
            return jsonify([{"Message": f"Param empty."}]), 500

        spacyNlp = SpacyNlpSentMessage(english_text)
        english = spacyNlp.NounChunk()
        
        spanishSpacyNlp = SpanishSpacyNlpSentMessage(spanish_text)
        spanish = spanishSpacyNlp.NounChunk()

        return jsonify([{"English": english, "Spanish": spanish}]), 200       
    except Exception as e:
        print({ "Message": f"Error: {str(e)}" })
        return jsonify({"Message": f"Error message, status code: 500"}), 500

@app.route("/nlp/generate_raw_sentiments")
def generate_raw_sentiments():
    try:
        english_text = unquote(request.args.get('english_text'))
        spanish_text = unquote(request.args.get('spanish_text'))

        if english_text == "empty" or spanish_text == "empty":
            return jsonify([{"Message": f"Param empty."}]), 500

        spacyNlp = SpacyNlpSentMessage(english_text)
        english = spacyNlp.Sentiments()
        
        spanishSpacyNlp = SpanishSpacyNlpSentMessage(spanish_text)
        spanish = spanishSpacyNlp.Sentiments()

        return jsonify([{"English": english, "Spanish": spanish}]), 200       
    except Exception as e:
        print({ "Message": f"Error: {str(e)}" })
        return jsonify({"Message": f"Error message, status code: 500"}), 500


@app.route("/nlp/generate_noun_chunks")
def generate_noun_chunks():
    try:
        title = request.args.get('title')
        searchString = request.args.get('searchString')
        text = request.args.get('text')
        file_path = r"C:\Users\dell\Entrepreneurship\Instructor\Algorithms\APIs\Web_API_Cloud_Services_Project.txt"
        doc = str(open(file_path, encoding='utf-8').read())
        spacyNlp = SpacyNlp(title, searchString, doc)
        data = spacyNlp.NounChunk()

        df = pd.DataFrame(data)
        df.to_csv(r"C:\Users\dell\Entrepreneurship\Engineering\ai_repo\Resume\noun_chunks.csv")

        return jsonify(data), 200       
    except Exception as e:
        print({ "Message": f"Error: {str(e)}" })
        return jsonify([{"Message": f"Error message, status code: 500"}]), 500

@app.route("/nlp/generate_entities")
def generate_entities():
    try:
        title = request.args.get('title')
        searchString = request.args.get('searchString')
        text = request.args.get('text')
        
        file_path = r"C:\Users\dell\Entrepreneurship\Instructor\Algorithms\APIs\Web_API_Cloud_Services_Project.txt"
        doc = str(open(file_path, encoding='utf-8').read())
        spacyNlp = SpacyNlp(title, searchString, doc)
        data = spacyNlp.EntityRecognition()

        df = pd.DataFrame(data)
        df.to_csv(r"C:\Users\dell\Entrepreneurship\Engineering\ai_repo\Resume\entities.csv")

        return jsonify(data), 200   
    except Exception as e:
        print({ "Message": f"Error: {str(e)}" })
        return jsonify([{"Message": f"Error message, status code: 500"}]), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5005)