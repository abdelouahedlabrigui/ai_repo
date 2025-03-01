import json
import spacy


class NERAndSyntacticParsing:
    def __init__(self, json_filename):
        self.json_filename = json_filename

    def ner_and_syntactic_parsing(self):
        nlp = spacy.load("en_core_web_sm")
        with open(self.json_filename, "r") as file:
            data = json.load(file)

        doc = nlp(data["original_text"])
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        syntactic_parse = [(token.text, token.dep_, token.head.text, token.head.pos_, [child.text for child in token.children]) for token in doc]
        data['entities'] = entities
        data['syntactic_parse'] = syntactic_parse
        return data

def main():
    preprocessed_json = r'C:\Users\dell\Entrepreneurship\Engineering\ai_repo\NLP\Preprocessed.json'
    ner_and_syntactic_parsing = NERAndSyntacticParsing(preprocessed_json).ner_and_syntactic_parsing()
    written_file = r'C:\Users\dell\Entrepreneurship\Engineering\ai_repo\NLP\NERAndSyntacticParsing.json'
    with open(f'{written_file}', 'w', encoding='utf-8') as file:
        file.write(json.dumps(ner_and_syntactic_parsing, indent=4))

if __name__ == "__main__":
    main()