class TextIntervals:
    def __init__(self, text_file: str, start_str: str, end_str: str):
        self.text_file = text_file
        self.start_str = start_str
        self.end_str = end_str

    def extract_text_between(self):
        with open(self.text_file, 'r', encoding='utf-8') as file:
            text = file.read()

        start_idx = text.find(self.start_str)
        end_idx = text.find(self.end_str, start_idx + len(self.start_str))

        if start_idx == -1 or end_idx == -1:
            return None
        
        return text[start_idx + len(self.start_str) : end_idx].strip()