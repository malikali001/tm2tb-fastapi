"""Extract terms from a sentence or multiple sentences."""

import os
import re
from collections import defaultdict
from typing import List
from functools import cached_property
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from spacy.tokens import Span
from spacy.matcher import Matcher
from tm2tb import trf_model
from tm2tb import get_spacy_model
from tm2tb.utils import detect_lang
from sklearn.cluster import KMeans


class TermExtractor:
    """Class representing a term extractor."""
    def __init__(self, texts: List[str], lang=None):
        self.texts = texts
        if lang is None:
            self.lang = detect_lang(self.texts)
        else:
            self.lang = lang
        Span.set_extension("similarity", default=None, force=True)
        Span.set_extension("rank", default=None, force=True)
        Span.set_extension("cluster", default=None, force=True)
        Span.set_extension("span_id", default=None, force=True)
        Span.set_extension("embedding", default=None, force=True)
        Span.set_extension("frequency", default=None, force=True)
        Span.set_extension("docs_idx", default=None, force=True)

    @cached_property
    def frequent_nouns(self):
        path = os.path.join('stopwords', f'{self.lang}_frequent_nouns.txt')
        with open(path, 'r', encoding='utf8') as fw:
            return fw.read().split('\n')

    @cached_property
    def frequent_adjs(self):
        path = os.path.join('stopwords', f'{self.lang}_frequent_adjs.txt')
        with open(path, 'r', encoding='utf8') as fw:
            return fw.read().split('\n')

    @cached_property
    def matcher(self):
        noun = {'POS': {'IN': ['NOUN']}, 'IS_PUNCT': False, 'LIKE_NUM': False}
        adj = {'POS': 'ADJ', 'IS_PUNCT': False, 'OP': '*'}
        adp = {'POS': 'ADP', 'IS_PUNCT': False}
        nVj = {'POS': {'IN': ['ADJ', 'NOUN']}, 'OP': '*', 'IS_PUNCT': False, 'LIKE_NUM': False}

        noun_final = [[nVj, noun], [nVj, noun, adp, nVj, noun]]
        noun_initial = [[noun, nVj], [noun, nVj, adp, noun, nVj]]

        PATTERNS = {
            'en': noun_final,
            'de': [[adj, noun], [noun, noun]],
            'fr': noun_initial,
            'es': noun_initial,
            'it': noun_initial,
            'pt': noun_initial
            }
        patterns = PATTERNS[self.lang]
        matcher = Matcher(self.nlp.vocab)
        for i, pattern in enumerate(patterns):
            matcher.add(f"{i}", [pattern])
        return matcher

    @cached_property
    def nlp(self):
        """Spacy lang model."""
        return get_spacy_model(self.lang)

    def extract_terms(self,
                      span_range=(1, 2),
                      freq_min=1,
                      term_length_min=2,
                      filter_stopwords=True,
                      include_entities=False):
        """
        Parameters
        ----------
        span_range : str, optional
            Length range of the terms. The default is (1, 2)
        freq_min : int, optional
            Minimum ocurrence frequency of the terms. The default is 1
        term_length_min : int, optional
            Minimum term length. The default is 2 characters.
        filter_stopwords : bool, optional
            If True, terms that contain a stopword are discarded. The default is True
        include_entities : bool, optional
            If True, the document entities are added to the terms results. The default is False
        Returns
        -------
        spans : List of spacy.tokens.span.Span objects
            A list of spans representing the terms from the document.
        """
        # Normalize whitespace
        texts = [' '.join(text.split()) for text in self.texts]
        # Get docs and doc embedding average
        docs = list(self.nlp.pipe(texts))
        docs_embeddings_avg = sum((trf_model.encode([doc.text for doc in docs]))/len(docs)).reshape(1, -1)

        # Collect spans, their frequencies and docs ids.
        spans = []
        spans_freqs_dict = defaultdict(int)
        spans_docs_dict = defaultdict(set)

        for doc_id, doc in enumerate(docs):

            if include_entities is True:
                stop_tags = {'DET', 'PUNCT', 'AUX', 'VERB', 'NUM'}
                stop_labels = {'ORDINAL', 'CARDINAL', 'TIME', 'DATE', 'QUANTITY', 'PERCENT', 'MONEY'}
                for ent in list(doc.ents):
                    # Disregard entities with determiners, punctuation, verbs and also numeric entities
                    if all(token.pos_ not in stop_tags for token in ent) and not ent.label_ in stop_labels:
                        if ent.text not in spans_freqs_dict.keys():
                            spans.append(ent)
                        spans_freqs_dict[ent.text] += 1
                        spans_docs_dict[ent.text].add(doc_id)

            # Select POS-pattern-matched spans
            matches = self.matcher(doc)
            for (_, start, end) in matches:
                span = doc[start:end]
                if span.text not in spans_freqs_dict.keys():
                    spans.append(span)
                spans_freqs_dict[span.text] += 1
                spans_docs_dict[span.text].add(doc_id)

        # Add frequency and doc id data to spans
        for span in spans:
            span._.frequency = spans_freqs_dict[span.text]
            span._.docs_idx = spans_docs_dict[span.text]

        # Use the passed parameters to filter the spans
        spans = filter(lambda term: len(term.text) >= term_length_min, spans)
        spans = filter(lambda term: term._.frequency >= freq_min, spans)
        spans = filter(lambda span: span_range[0] <= len(span) <= span_range[1], spans)
        spans = list(spans)
        if filter_stopwords is True:
            spans = self._filter_stopwords(spans)

        # Trim spans
        spans = self._trim_spans(spans, texts)

        # Get spans embeddings
        spans_embeddings = trf_model.encode([s.text for s in spans])

        # Get doc/spans similarities
        spans_doc_similarities = cosine_similarity(spans_embeddings, docs_embeddings_avg)
        for idx, span in enumerate(spans):
            span._.similarity = round(float(spans_doc_similarities.reshape(1, -1)[0][idx]), 4)
            span._.embedding = spans_embeddings[idx]

        # Rank spans
        spans = self._simple_rank(spans)

        # Cluster spans
        spans = self._cluster_spans(spans)
        
        # Assign span id and return spans
        for span_id, span in enumerate(spans):
            span._.span_id = span_id

        return spans

    def _cluster_spans(self, spans):
        n_clusters = round(len(spans)*.3)
        embs = [span._.embedding for span in spans]
        cluster_labels = KMeans(n_clusters=n_clusters, random_state=0).fit(embs).labels_
        for idx, label in enumerate(cluster_labels):
            spans[idx]._.cluster = label
        return spans

    def _filter_stopwords(self, spans):
        # Keep spans if none of its (lower-cased) tokens' lemmas is in stopwords.
        spans_ = []
        for span in spans:
            # Filter if span is non-entity
            if span.label_ == '':
                # filter if term lemma is in frequent nouns or if any frequent adjective is in the lemmatized term
                if span.lemma_.lower() not in self.frequent_nouns \
                        and not set([tok.lemma_.lower() for tok in span]) & set(self.frequent_adjs):
                    spans_.append(span)
            else:
                spans_.append(span)
        if len(spans_) == 0:
            raise ValueError('No terms found.')
        return spans_

    def _trim_spans(self, spans, texts):
        # When a term is a subset of another term for any given "span", only keep the longer term
        spans_indices =  [[m.start(0) for m in re.finditer(span.text, texts[0])] for span in spans]
        term_lens = [len(span.text) for span in spans]
        longest_match_span = max(term_lens)
        candidates = [m for i, m in enumerate(spans) if term_lens[i] < longest_match_span]
        match_spans = set()
        for i, matches in enumerate(spans_indices):
            for idx in matches:
                match_spans.add(tuple(range(idx, idx + term_lens[i])))
        for cand in candidates:
            idx = spans.index(cand)
            cand_spans = [tuple(range(x, x + term_lens[idx])) for x in spans_indices[idx]]
            for cand_span in cand_spans:
                for match_span in match_spans - {cand_span}:
                    if all(x in match_span for x in cand_span):
                        try:
                            spans_indices[idx].remove(cand_span[0])
                        except ValueError:
                            pass
        return [spans[i] for i in range(len(spans)) if len(spans_indices[i])>0]

    @staticmethod
    def _simple_rank(spans):
        # Define rank as term-to-doc similarity plus normalized term frequency
        for span in spans:
            span._.rank = span._.similarity * 1/(1 + np.exp(-span._.frequency))
        return spans

    @staticmethod
    def _mmr_rank(spans, spans_doc_sims):
        # Rank terms using Maximal Marginal Relevance
        top_n = round(len(spans)/2)
        diversity = .9
        spans_sims = cosine_similarity([sp._.embedding for sp in spans])
        best_spans_idx = [np.argmax(spans_doc_sims)]
        candidates_idx = [i for i in range(len(spans)) if i != best_spans_idx[0]]
        for _ in range(min(top_n - 1, len(spans) - 1)):
            candidate_sims = spans_doc_sims[candidates_idx, :]
            rest_spans_sims = np.max(spans_sims[candidates_idx][:, best_spans_idx], axis=1)
            # Calculate Maximum Marginal Relevance
            mmr = (1-diversity) * candidate_sims - diversity * rest_spans_sims.reshape(-1, 1)
            # Get best candidate
            mmr_idx = candidates_idx[np.argmax(mmr)]
            # Update best spans & candidates
            best_spans_idx.append(mmr_idx)
            candidates_idx.remove(mmr_idx)

        # Add rank to spans
        for span in spans:
            idx = spans.index(span)
            if idx in best_spans_idx:
                span = spans[idx]
                span._.rank = span._.similarity * 1/(1 + np.exp(-span._.frequency))
            if idx in candidates_idx:
                span = spans[idx]
                span._.rank = (span._.similarity * 1/(1 + np.exp(-span._.frequency))) * .5
        spans = sorted(spans, key=lambda span: span._.rank, reverse=True)
        return spans
