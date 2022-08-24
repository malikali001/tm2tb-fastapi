"""
TM2TB initialization
"""
from tm2tb.transformer_model import TransformerModel

trf_model = TransformerModel("LaBSE").load()

from tm2tb.spacy_models import get_spacy_model
from tm2tb.term_extractor import TermExtractor
from tm2tb.biterm_extractor import BitermExtractor

#import nltk
#nltk.download('punkt')
from tm2tb.summarizer import Summarizer
