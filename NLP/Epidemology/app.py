from flask import Flask, json, jsonify, request
from urllib.parse import unquote
from TextIntervals.TextIntervals import TextIntervals
from TopicCategory.TopicCategory import TopicCategory
from Classification.Classification import Classification
import os.path

app = Flask(__name__)

@app.route('/topics/extract_text_between', methods=['GET'])
def extract_text_between():
    try:
        text_file = unquote(request.args.get("text_file"))
        start_str = unquote(request.args.get("start_str"))
        end_str = unquote(request.args.get("end_str"))

        if os.path.isfile(text_file) == False:
            return jsonify({"Message": "File not found"}), 400
        
        extract_text = TextIntervals(text_file=text_file, start_str=start_str, end_str=end_str)
        text = extract_text.extract_text_between()

        if len(text) == 0:
            return jsonify({"Message": f"Empty text with length {len(text)}"}), 400
        
        topic = TopicCategory(text=text)
        data = topic.sentence_clustering()
        
        if len(data) == 0:
            return jsonify({"Message": f"Data length is: {len(data)}"}), 400

        return jsonify(data), 200
    except Exception as e:
        return jsonify({"Message": str(e)}), 500
    
@app.route('/nlp/text_classification', methods=["GET"])
def text_classification():
    try:
        text_file = unquote(request.args.get("text_file"))
        start_str = unquote(request.args.get("start_str"))
        end_str = unquote(request.args.get("end_str"))

        if os.path.isfile(text_file) == False:
            return jsonify({"Message": "File not found"}), 400
        
        classifier = Classification(text_file=text_file, start_str=start_str, end_str=end_str)

        return jsonify({"Message": "Ok."}), 200
    except Exception as e:
        return jsonify({"Message": str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True, port=5005)