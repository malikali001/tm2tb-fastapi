from langdetect import detect, LangDetectException
from collections import Counter as cnt
from typing import Union
from random import randint
import re

pattern = re.compile(r"(.)(\n|\\n|\\\n|\\\\n|\\\\\n)(.)")


def detect_lang(input_: Union[str, list]):
    """
    This method uses langdetect to detect the language of one or more sentences.
    (See https://pypi.org/project/langdetect/)
    The languages supported depends on the spaCy language modules installed.

    For one sentence, it simply passes the sentence to the langdetect module.
    For multiple sentences, it takes the most common language detected from a sample of sentences.
    This is done to speed-up the process.

    Parameters
    ----------
    input_ : Union[str, list]
        DESCRIPTION. String or list of strings

    Raises
    ------
    ValueError
        DESCRIPTION. Error raised when the language identified is not supported.

    Returns
    -------
    lang : string
        DESCRIPTION. Two-character language identifier.

    """
    supported_languages = {'en', 'es', 'de', 'fr', 'it', 'pt'}
    if isinstance(input_, str):
        lang = detect(input_)
        if lang not in supported_languages:
            raise ValueError('Language not supported!')
    if isinstance(input_, list):
        text = input_
        if len(text) <= 50:
            text_sample = text
        else:
            rand_start = randint(0, (len(text) - 1) - 50)
            text_sample = text[rand_start:rand_start + 50]
        detections = []

        for sentence in text_sample:
            try:
                detections.append(detect(sentence))
            except LangDetectException:
                pass
        if len(detections) == 0:
            raise ValueError('Insufficient data to detect language')
        lang = cnt(detections).most_common(1)[0][0]
        if lang not in supported_languages:
            raise ValueError('Language not supported!')
    return lang


def preprocess(sentence):
    """
    Minimal preprocessing function.
    It just normalizes spaces and new line characters.

    Parameters
    ----------
    sentence : string
        DESCRIPTION. String representing one sentence or short paragraph.

    Returns
    -------
    string
        DESCRIPTION. The same string, but minimally cleaned.

    """

    def normalize_space_chars(sentence):
        """
        Replaces all spaces with normal spaces.
        """
        chars = [chr(x) for x in [9, 10, 13, 32, 160]]
        for char in chars:
            sentence = sentence.replace(char, ' ')
        return sentence

    def normalize_space_seqs(sentence):
        """
        Finds sequences of more than one space, returns one space.
        """
        sentence = ' '.join(sentence.split())
        return sentence

    def normalize_newline(sentence):
        """
        Replaces hard coded newlines with normal newline symbol.
        """

        def repl(sentence):
            groups = sentence.groups()
            return f'{groups[0]}\n{groups[2]}'

        return re.sub(pattern, repl, sentence)

    return normalize_newline(normalize_space_seqs(normalize_space_chars(sentence)))
