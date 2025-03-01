import nltk
import json

class PosTagging:
    def __init__(self, json_file):
        self.json_file = json_file

    def pos_tagging(self):
        with open(f'{self.json_file}', 'r', encoding='utf-8') as file:
            preprocessed_data = json.load(file)

        nltk.download('averaged_perceptron_tagger')
        nltk.download('averaged_perceptron_tagger_eng')
        pos_tags = nltk.pos_tag(preprocessed_data['filtered_words'])

        preprocessed_data['pos_tags'] = pos_tags    

        return preprocessed_data
    
def main():
    json_file = r'C:\Users\dell\Entrepreneurship\Engineering\ai_repo\NLP\Preprocessed.json'
    write_json_tagging_file = r'C:\Users\dell\Entrepreneurship\Engineering\ai_repo\NLP\PosTaggedPreprocessed.json'
    pos_tagger = PosTagging(json_file)  
    tagged_data = json.dumps(pos_tagger.pos_tagging(), indent=4)
    with open(f'{write_json_tagging_file}', 'w', encoding='utf-8') as file:
        file.write(tagged_data)

if __name__ == '__main__':
    main()