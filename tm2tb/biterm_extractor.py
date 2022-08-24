"""
Extract bilingual terms from one or multiple sentences.

Classes:
    BitermExtractor

Functions:
    extract_terms(List[str], **kwargs)
"""
import requests
import uuid
from collections import defaultdict
from collections import namedtuple
from typing import List
from itertools import groupby
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tm2tb import TermExtractor


class BitermExtractor:
    """

    Class representing a bilingual term extractor.

    Attributes
    ----------
    input_ : List[str]
        DESCRIPTION. List of strings

    Methods
    -------
    extract_terms(self, **kwargs)

    """

    def __init__(self, input_: List[str], src_lang=None, tgt_lang=None):
        self.input_ = input_
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def extract_terms(self,
                      similarity_min=.9,
                      return_unmatched_terms=True,
                      mt_unmatched_terms=False,
                      collapse_lemmas=False,
                      **kwargs):
        """
        Extract biterms from an unaligned pair of a source text and a target text.

        Compare the embeddings of both terms lists and get the best matches.
        The best target term for each source term (and viceversa) are retrieved.

        Parameters
        ----------
        similarity_min : float, optional
            Minimum similarity value between source and target terms.
            The default is .9.
        return_unmatched_terms : bool, optional
            If True, src_terms without a match are returned
        collapse_lemmas : bool, optional
            If True, the list of resulting terms is collapsed using their lemmas. The default is False
        **kwargs : dict
            See the parameters accepted by TermExtractor.

        Returns
        -------
        terms : list
            Pandas dataframe representing biterms and their metadata.

        """
        src_texts, tgt_texts = zip(*self.input_)

        # Get source terms
        src_extractor = TermExtractor(list(src_texts), lang=self.src_lang)
        src_terms = src_extractor.extract_terms(**kwargs)

        # Get target terms
        tgt_extractor = TermExtractor(list(tgt_texts), lang=self.tgt_lang)
        tgt_terms = tgt_extractor.extract_terms(**kwargs)

        # Get source/target terms similarities
        sm = self._get_similarity_matrix(src_terms, tgt_terms)

        # Get the top biterm matches from the similarity matrix
        biterms_idx = list(
            zip(range(len(src_terms)), np.argmax(sm, axis=1), np.max(sm, axis=1)))
        biterms_matches = [(list(src_terms)[i], list(tgt_terms)[i_], n)
                           for (i, i_, n) in biterms_idx]

        # Collect biterm frequencies, similarities and strings
        biterms_freqs_dict = defaultdict(int)
        biterms_sims_dict = defaultdict(set)
        biterms_strings_dict = defaultdict(set)
        for src_term, tgt_term, biterm_similarity in biterms_matches:
            biterms_strings_dict[(src_term.text, tgt_term.text)] = (
                src_term, tgt_term)
            biterms_sims_dict[(src_term.text, tgt_term.text)
                              ] = round(biterm_similarity, 4)
            biterms_freqs_dict[(src_term.text, tgt_term.text)] += 1

        # Build, prune and filter biterms
        BiTerm = namedtuple('BiTerm', ['src_term', 'src_tags', 'src_rank',
                                       'src_label', 'src_frequency', 'src_cluster',
                                       'tgt_term', 'tgt_tags', 'tgt_rank',
                                       'similarity', 'frequency', 'biterm_rank', 'origin'])
        biterms = self._build_biterms(BiTerm, biterms_freqs_dict, biterms_sims_dict,
                                      biterms_strings_dict, similarity_min)
        biterms = self._prune_biterms(biterms, 'src')
        biterms = self._prune_biterms(biterms, 'tgt')
        if return_unmatched_terms is True:
            biterms, biterms_strings_dict = self._return_unmatched_terms(BiTerm, src_terms,
                                                                         biterms, biterms_strings_dict)
        if collapse_lemmas is True:
            biterms = self._collapse_biterm_lemmas(
                biterms, biterms_strings_dict)
        if return_unmatched_terms is True and mt_unmatched_terms is True:
            biterms = self._mt_unmatched_terms(biterms)
        return self._return_as_df(biterms)

    @staticmethod
    def _get_similarity_matrix(src_spans, tgt_spans):
        """
        Generate a similarity matrix of source and target term candidates.

        Parameters
        ----------
        src_spans : list
            List of spans of type 'spacy.tokens.span.Span' from the source side
        tgt_spans : list
            List of spans of type 'spacy.tokens.span.Span' from the target side

        Returns
        -------
        similarity_matrix : numpy.ndarray
            Similarity matrix produced by calculating the cosine similarity
            of the source and the target spans embeddings.

        """
        src_embeddings = [span._.embedding for span in src_spans]
        tgt_embeddings = [span._.embedding for span in tgt_spans]
        similarity_matrix = cosine_similarity(src_embeddings, tgt_embeddings)
        return similarity_matrix

    @staticmethod
    def _build_biterms(BiTerm, biterms_freqs_dict, biterms_sims_dict, biterms_strings_dict, similarity_min):
        """
        Gather the metadata from the biterms and build the final biterms list.

        Parameters
        ----------
        biterms_freqs_dict: dict representing biterms frequencies
        biterms_sims_dict: dict representing biterms similarities
        biterms_strings_dict: dict representing biterms strings

        Returns
        -------
        biterms : list
            List of named tuples representing the extracted bilingual terms.
        """
        biterms = []
        for biterm_, similarity in biterms_sims_dict.items():
            if similarity >= similarity_min:
                src_term = biterm_[0]
                tgt_term = biterm_[1]
                spans = biterms_strings_dict[biterm_]
                src_span = spans[0]
                tgt_span = spans[1]
                src_tags = [t.pos_ for t in src_span]
                tgt_tags = [t.pos_ for t in tgt_span]
                src_rank = src_span._.rank
                tgt_rank = tgt_span._.rank
                src_label = src_span.label_
                src_frequency = src_span._.frequency
                src_cluster = src_span._.cluster
                frequency = biterms_freqs_dict[biterm_]
                biterm_rank = biterm_rank = biterm_rank = (
                    (src_rank + tgt_rank) / 2) * similarity
                origin = 'similarity'
                biterm = BiTerm(src_term, src_tags, src_rank,
                                src_label, src_frequency, src_cluster,
                                tgt_term, tgt_tags, tgt_rank,
                                similarity, frequency, biterm_rank, origin)
                biterms.append(biterm)
        if len(biterms) == 0:
            raise ValueError('No biterms found.')
        return biterms

    @staticmethod
    def _prune_biterms(biterms, side):
        """
        Keep the most similar target term for each source term (and viceversa).

        Parameters
        ----------
        biterms : list
            List of named tuples.
        side : str
            String representing which "side" to prune: "src" or "tgt".

        Returns
        -------
        biterms_ : list
            Pruned list of named tuples in the same format.
        """
        if side == 'src':
            def keyfunc(k): return k.src_term
            biterms = sorted(biterms, key=keyfunc)
        else:
            def keyfunc(k): return k.tgt_term
            biterms = sorted(biterms, key=keyfunc)

        biterms_ = []
        for _, group in groupby(biterms, keyfunc):
            # Sort biterms group by similarity
            group = sorted(group,
                           key=lambda group: group.similarity,
                           reverse=True)
            # Take first biterm
            best_biterm = list(group)[0]
            biterms_.append(best_biterm)
        return biterms_

    @staticmethod
    def _return_unmatched_terms(BiTerm, src_terms, biterms, biterms_strings_dict):
        """Add unmatched source terms to biterms."""
        for span in src_terms:
            if span.text not in [biterms[n].src_term for n in range(len(biterms))]:
                src_term = span.text
                src_tags = [t.pos_ for t in span]
                src_rank = span._.rank
                src_label = span.label_
                src_frequency = span._.frequency
                src_cluster = span._.cluster
                tgt_term = ''
                tgt_rank = 0
                tgt_tags = ['']
                frequency = 0
                similarity = 0
                biterm_rank = 0
                origin = ''
                biterm = BiTerm(src_term, src_tags, src_rank,
                                src_label, src_frequency, src_cluster,
                                tgt_term, tgt_tags, tgt_rank,
                                similarity, frequency, biterm_rank, origin)
                biterms_strings_dict[(src_term, tgt_term)] = (span, tgt_term)
                biterms.append(biterm)
        return ((biterms, biterms_strings_dict))

    @staticmethod
    def _collapse_biterm_lemmas(biterms, biterms_strings_dict):
        biterms_ = []
        seen_src_lemmas = []
        # Sort biterms by rank
        for biterm in sorted(biterms, key=lambda biterm: biterm.biterm_rank, reverse=True):
            # Retrieve the source term (as spaCy Span, so that we can check its lemma and label.)
            src_span = biterms_strings_dict[(
                biterm.src_term, biterm.tgt_term)][0]
            # If the lower-cased lemmatized source term has not been seen, and is not an entity, keep the biterm
            if src_span.label_ == '':
                if src_span.lemma_.lower() not in seen_src_lemmas:
                    biterms_.append(biterm)
                seen_src_lemmas.append(src_span.lemma_.lower())
            else:
                biterms_.append(biterm)
        if len(biterms_) == 0:
            raise ValueError('No biterms found.')
        return biterms_

    def _mt_unmatched_terms(self, biterms):
        # MT API config
        translation_url = 'https://api.cognitive.microsofttranslator.com'
        translation_key = '830a4539e6914c8ea68e4b912ca678af'
        translation_region = 'eastus'
        path = '/translate'
        constructed_url = translation_url + path
        params = {'api-version': '3.0',
                  'from': self.src_lang, 'to': self.tgt_lang}
        headers = {'Ocp-Apim-Subscription-Key': translation_key,
                   'Ocp-Apim-Subscription-Region': translation_region,
                   'Content-type': 'application/json',
                   'X-ClientTraceId': str(uuid.uuid4())}

        # MT unmatched src terms
        src_terms_unm = [{'text': biterm.src_term}
                         for biterm in biterms if biterm.tgt_term == '']
        if len(src_terms_unm) > 0:
            request = requests.post(
                constructed_url, params=params, headers=headers, json=src_terms_unm)
            data = request.json()

            # Replace empty tgt terms with MT terms
            tgt_idx = 0
            for idx, biterm in enumerate(biterms):
                if biterms[idx].tgt_term == '':
                    tgt_term_mt = data[tgt_idx]['translations'][0]['text']
                    biterms[idx] = biterms[idx]._replace(tgt_term=tgt_term_mt)
                    biterms[idx] = biterms[idx]._replace(origin='MT')
                    tgt_idx += 1
        return biterms

    @staticmethod
    def _return_as_df(biterms):
        """Return biterms as pandas dataframe."""
        biterms = pd.DataFrame(biterms)
        col_names = ['src_terms', 'src_tags', 'src_ranks',
                     'src_labels', 'src_frequencies', 'src_clusters',
                     'tgt_terms', 'tgt_tags', 'tgt_ranks',
                     'similarities', 'frequencies', 'ranks', 'origins']
        biterms.columns = col_names
        biterms.reset_index(drop=True, inplace=True)
        biterms['similarities'] = biterms['similarities'].astype(
            float).round(3)
        biterms['ranks'] = biterms['ranks'].astype(float).round(3)
        biterms = biterms.drop(
            columns=['src_ranks', 'src_tags', 'tgt_ranks', 'tgt_tags'])
        return biterms.sort_values(by='ranks', ascending=False)
