"""
Summarizer.
Implements a method to extract a summary from a text.
"""
import os
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# Add nltk_data path
nltk_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_path)

class Summarizer:
    def __init__(self, texts, lang, summary_sentences_n):
        self.texts = texts
        self.lang = lang
        self.summary_sentences_n = summary_sentences_n

    def extract_summary(self):
        # Create text parser using tokenization
        texts = ' '.join(self.texts)
        parser = PlaintextParser.from_string(texts, Tokenizer(self.lang))

        # Summarize using sumy TextRank
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, self.summary_sentences_n)

        text_summary = [str(sentence) for sentence in summary]
        return text_summary
