import pandas as pd
from Transformers.BridgingTheGap import TextClassification
from Transformers.BridgingTheGap import NerTagger
from Transformers.BridgingTheGap import QuestionAnswering
from Transformers.BridgingTheGap import Summarization
from Transformers.BridgingTheGap import Translation
from Transformers.BridgingTheGap import TextGeneration
from flask import Flask, json, jsonify, request
from urllib.parse import unquote

app = Flask(__name__)

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
    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5005)