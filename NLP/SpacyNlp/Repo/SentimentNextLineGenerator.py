import json
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from sklearn.metrics import accuracy_score, precision_score
import torch
import spacy
nlp = spacy.load("en_core_web_sm")

class SentimentNextLineGenerator:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()   

        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_next_line(self, paragraph, sentiment, max_new_tokens=50, temperature=0.7):
        if sentiment not in ['positive', 'negative']:
            raise ValueError("Sentiment must be either 'positive' or 'negative'")
        
        sentiment_prefix = "The sentiment of this paragraph is positive. " if sentiment == 'positive' else "The sentiment of this paragraph is negative. "
        prompt = sentiment_prefix + paragraph

        encoded_input = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=1024)
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']    

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        next_line = generated_text[len(prompt):].strip()

        return next_line    
    
    def transform_sentences(self, paragraph):
        data = []
        for sentence in str(paragraph).split('.'):
            sentence = sentence.strip()
            if sentence:
                positive_line = self.generate_next_line(paragraph, 'positive', max_new_tokens=50)
                negative_line = self.generate_next_line(paragraph, 'negative', max_new_tokens=50)
                data.append({
                    'Sentence': sentence,
                    'PositiveLine': positive_line,
                    'NegativeLine': negative_line
                })
        df = pd.DataFrame(data)
        df = df.drop_duplicates()
        return df
# Example Usage
if __name__ == "__main__":
    generator = SentimentNextLineGenerator()

    # Test dynamic paragraph length
    paragraph = "The RF semiconductor technology versus uses case shows a selection of typical system parameters relevant for semiconductors. Figure 3: Semiconductor technology overview, trends towards 2020 and use cases mm-Wave Semiconductor Industry Technologies - Status and Evolution 24 Baseband analogue frontend (AFE) technology overview Introduction Millimetre-wave modems utilise digital signal processing to perform complex (IQ) modulation and demodulation in order to transmit gigabit rate data streams onto radio carriers in V, E and other bands. Data rates range from 1 – 10 Gbps over channel bandwidths of 250, 500 MHz and more recently use of wider channel widths such 1 and 2 GHz. Modulation levels range from BPSK, QPSK, 16QAM and through to 256QAM for high-performance channel limited systems."
    df = generator.transform_sentences(paragraph)
    for sentence, positive_line, negative_line in zip(df['Sentence'], df['PositiveLine'], df['NegativeLine']):
        print(f"Sentence: {sentence}\nPositive Line: {positive_line}\nNegative Line: {negative_line}\n")
