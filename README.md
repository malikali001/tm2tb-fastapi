# tm2tb
# TM2TB API for KUDO

## Scope

The API receives a transcription in 2 languages (original and target language) and produces as output:

- a bilingual glossary with rank values
- a summary of the original
- a list of named entities

The bilingual glossary is produced by extracting terminology in the original and in translation and finding semantic similarities between the extracted words. Should an extracted term in original do not have a translation in the taarget language transcription, machine translation will be used.

## Installation

1. Clone the repository and enter the directory:

```bash
git clone https://github.com/kudohq/tm2tb
cd tm2tb
```

2. Start pipenv shell and install requirements

```bash
pipenv shell
pipenv install
```

3. You can now run locally within pipenv or you can exit pipenv run in docker:

### Running locally:

Ensure that the pipenv shell is active, then run:

```bash
uvicorn --reload main:app
```

### Running migrations:
    ```bash
    alembic upgrade head
    ```

### Running in docker:

Create the docker image & run it with port forwarding:

```bash
docker build -t tm2tb-api .
docker run -p 8000:8000 --detach tm2tb-api
```

This will allow you to access the API at http://localhost:8000/biterms, and you can view the API docs at http://localhost:8000/redoc.

There are command-line scripts available for making API calls. Usage is available by invoking the scripts with the `-h` option.

## Request and response data

The API takes a request representing bilingual sentences and their language IDs:

```
{'src_lang': 'it',
 'src_texts': ['Il panda gigante o panda maggiore è un mammifero appartenente '
               'alla famiglia degli orsi.',
               'Originario della Cina centrale, vive nelle regioni montuose '
               'del Sichuan.'],
 'tgt_lang': 'en',
 'tgt_texts': ['The giant panda or big panda is a mammal that belongs to the '
               'bear family.',
               'Native to central China, it lives in the mountainous regions '
               'of Sichuan.']}
```

If successful, the response of the API is a dict representing bilingual terms and their metadata.

```
{'ranks': [0.5557, 0.5305, 0.5233, 0.5226],
 'frequencies': [1, 1, 1, 1],
 'similarities': [0.9872, 0.9878, 1.0, 0.9895],
 'src_terms': ['panda gigante', 'Cina centrale', 'Sichuan', 'famiglia'],
 'tgt_terms': ['giant panda', 'central China', 'Sichuan', 'family']}
```

## Back-end parameters and attributes

For biterm extraction, the parameters that can be passed to the `BitermExtractor` are the following:

- `src_texts`: List of source sentences.

- `src_lang`: Two code language ID for the source language.
  (Optional. If not passed, the language is detected automatically.)
- `tgt_texts`: List of target sentences.
- `tgt_lang`: Two code language ID for the target language. (Optional. If not passed, the language is detected automatically.)
- `freq_min`: Minimum occurrence frequency of the source/target term pair. (Optional, the default is 1).
- `similarity_min`: Minimum similarity value (between 0 and 1) of the source/target term pair. The default is .9 (that is, biterms with a similarity below .9 are discarded.)
- `span_range`: Minimum and maximum length of patterns to match.
- `filter_stopwords`: Boolean value to filter stopwords or not. The default is `True`.
- `include_entities`: Boolean value to include entity spans or not. The default is `False`.

The resulting `biterms` object is a pandas dataframe with the following column names:

- `src_terms`: A list of terms of the source text.

- `src_terms_labels`: A label for the span representing an entity type.
- `src_tags`: A list of part-of-speech tags of each source term.
- `src_ranks`: The rank of each source term.
- `tgt_terms`: A list of terms of the target text (co-indexed with the src_terms).
- `tgt_tags`: A list of part-of-speech tags of each target term.
- `tgt_ranks`: The rank of each target term.
- `similarities`: The similarity measure of each source/target term pair.
- `frequencies`: The occurrence frequency of each source/target term pair.
- `ranks`: The rank of each source/target term pair.

For summarization, the accepted parameters are the following:

- `texts`: List of sentences.
- `lang`: Two code language ID.
- `summary_sentences_n`: Optional number of summary sentences to return

The summary response is a dictionary with this shape:

- {`summary`: [sentence1, sentence2, sentence3, ...]}

## Bilingual term extraction overview

The tm2tb module performs the term extraction in two steps:

### 1. Term extraction

    Example:
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "

        The string is tokenized:
            tokens:       ["Lore", "ipsum", "dolor", "..."]

        From the tokens, candidate terms are selected:
            candidate terms: 	["Lore ipsum", "dolor", "..."]

        Finally, the candidate terms are ranked:
            ranked terms:       [("Lore ipsum", 0.03), ("dolor", 0.02), ...]

    Terms are extracted from the source and the the target texts independently.

        [source texts]			[target texts]
              ↓                 ↓
        [source terms]   [target terms]

See a diagram of this process [here](https://github.com/fantinuoli/tm2tb/blob/main/monolingual_workflow.png).

### 2. Term alignment

    The second step is term alignment.

    The similarity of all source terms against all target terms is calculated.
    The result of this calculation is a similarity matrix, for example:

![](https://raw.githubusercontent.com/luismond/tm2tb/main/.gitignore/max_seq_similarities_small.png)

    For each source term, the most similar target term is retrieved from the matrix.
    This is how the collection of bilingual terms is created.

    Finally, other data such as frequency and rank is added to the biterms.

### 3. Machine Translation

    If a source term (original language) was not aligned with a translation from the target language, the term will be translated by means of Machine    Translation (Azure Machine Translation)
