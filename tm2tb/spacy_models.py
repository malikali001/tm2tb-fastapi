import spacy
"""
spaCy model selection.

TM2TB comes with 4 spaCy language models (English, Spanish, German and French).

In order to support additional languages,
the corresponding spaCy model must be installed.
Check the available spaCy language models here: https://spacy.io/models
"""

# Disable unneeded pipeline components
disabled_comps = ['entity_linker', 'trf_data', 'textcat']
print('Loading spacy models...')

models = {'en': spacy.load("en_core_web_md", exclude=disabled_comps),
          'de': spacy.load("de_core_news_md", exclude=disabled_comps),
          'es': spacy.load("es_core_news_md", exclude=disabled_comps),
          'fr': spacy.load("fr_core_news_md", exclude=disabled_comps),
          'it': spacy.load("it_core_news_md", exclude=disabled_comps),
          'pt': spacy.load("pt_core_news_md", exclude=disabled_comps)}


def get_spacy_model(lang):
    """
    Get spaCy model from one of the supported languages.

    Parameters
    ----------
    lang : string
        Two-character language identifier ('en', 'es', 'de', 'fr', 'it' or 'pt')

    Raises
    ------
    ValueError
        If no installed language models are found.

    Returns
    -------
    spacy_model : one of the supported spaCy models

        DESCRIPTION. spaCy language model
    """
    try:
        spacy_model = models[lang]
    except KeyError:
        raise ValueError(f"Model {lang}_core_news_md is not currently supported.")
    return spacy_model
