from argparse import ArgumentParser
import logging

import requests
from bs4 import BeautifulSoup as BSoup

FORMAT = '%(asctime)s\t%(message)s'
DATEFMT = '%H:%M:%S'


if __name__ == '__main__':
    parser = ArgumentParser(description='Script for simulating production environment to test different time variables')
    parser.add_argument('-u', '--url', dest='url', default='http://localhost:8000/v1_0/biterms',
                        help='URL to use for API calls')
    parser.add_argument('-s', '--srcfile', dest='source_file', required=True, help='hypertext source file')
    parser.add_argument('-t', '--tgtfile', dest='target_file', required=True, help='hypertext target file')
    parser.add_argument('--l1', dest='src_lang', default='en', help='2-letter language code for source')
    parser.add_argument('--l2', dest='tgt_lang', required=True, help='2-letter language code for target')
    parser.add_argument('-w', '--window', type=int, dest='window', default=20000, help='window size in milliseconds')
    parser.add_argument('-d', '--delay', type=int, dest='delay', default=2000,
                        help='delay between source & target windows in milliseconds')
    parser.add_argument('-g', '--logfile', dest='logfile', default='streaming.log', help='file to log results to')
    args = parser.parse_args()

    # start logging
    logging.basicConfig(filename=args.logfile, format=FORMAT, datefmt=DATEFMT, encoding='utf-8', level=logging.INFO)

    # open & parse hypertext files
    logging.info(f'Reading {args.source_file}')
    with open(args.source_file, encoding='utf-8') as fh:
        src_soup = BSoup(fh.read(), 'html.parser')

    logging.info(f'Parsing {args.source_file} data')
    spans = src_soup.find_all('span')
    src_words = [x.text.strip() for x in spans]
    src_word_offsets = [int(x['data-m']) for x in spans]

    logging.info(f'Reading {args.target_file}')
    with open(args.target_file, encoding='utf-8') as fh:
        tgt_soup = BSoup(fh.read(), 'html.parser')

    logging.info(f'Parsing {args.source_file} data')
    spans = tgt_soup.find_all('span')
    tgt_words = [x.text.strip() for x in spans]
    tgt_word_offsets = [int(x['data-m']) for x in spans]

    logging.info(f'Simulating {args.window}ms window with {args.delay}ms interpretation delay...')
    for i, src_offset in enumerate(src_word_offsets):
        src_span = [src_words[i]]
        for j, cand in enumerate(src_word_offsets):
            if src_offset < cand <= src_offset + args.window:
                src_span.append(src_words[j])
            elif cand > src_offset + args.window:
                break
        tgt_span = []
        for j, cand in enumerate(tgt_word_offsets):
            if src_offset + args.delay <= cand <= src_offset + args.window + args.delay:
                tgt_span.append(tgt_words[j])
            elif cand > src_offset + args.window + args.delay:
                break
        if len(src_span) > 0 and len(tgt_span) > 0:
            src_text = ' '.join(src_span)
            tgt_text = ' '.join(tgt_span)
            pload = {'src_texts': [src_text], 'tgt_texts': [tgt_text],
                     'src_lang': args.src_lang, 'tgt_lang': args.tgt_lang}
            logging.info('Sending texts to API...')
            logging.info(f'...source: {src_text}')
            logging.info(f'...target: {tgt_text}')
            r = requests.post(args.url, json=pload)
            if r.ok:
                logging.info('Response ok! Terms:')
                tmp = r.json()
                for j, src_term in enumerate(tmp['src_terms']):
                    tgt_term = tmp['tgt_terms'][j]
                    similarity = tmp['similarities'][j]
                    frequency = tmp['frequencies'][j]
                    rank = tmp['ranks'][j]
                    logging.info(f'\t{src_term}\t{tgt_term}\t{similarity}\t{frequency}\t{rank}')
            else:
                logging.error(f'Problem with API call: {r.text}')

    logging.info('Simulation complete')
