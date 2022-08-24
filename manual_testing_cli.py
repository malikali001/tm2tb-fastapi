"""Manual testing CLI."""

import sys
from multiprocessing import Pool
from argparse import ArgumentParser
import requests


def get_biterms_from_api(args, line):
    src_text, tgt_text, src_lang, tgt_lang = line.split("\t")

    pload = {"src_texts": [src_text], "tgt_texts": [tgt_text],
             "src_lang": src_lang, "tgt_lang": tgt_lang,
             "similarity_min": args.similarity_min,
             "collapse_lemmas": True}

    response = requests.post(url=args.url, headers={'access_token': 'fb5169d5dc7149d4bff96674c67f1f10'}, json=pload)
    if response.ok:
        result = response.json()
    else:
        result = {"error": response}
    return result


if __name__ == "__main__":
    parser = ArgumentParser("Run batch bitext term extraction requests.")
    parser.add_argument("-u", "--url", dest="url", default="http://localhost:8000/v1_0/biterms",
                        help="URL to use for API calls")
    parser.add_argument("-s", "--similarity_min", dest="similarity_min", default=.8,
                        help="Minimum similarity value of src/tgt term pairs")
    parser.add_argument("-o", "--outfile", dest="output_file", required=True,
                        help="base name of file to save output in .tsv format")
    parser.add_argument(dest="input_file", help=(".tsv file with source text, target text, "
                                                 "source language and target language per line"))
    args = parser.parse_args()

    if not args.input_file.endswith(".tsv"):
        print(f"Input file {args.input_file} does not match .tsv")
        sys.exit(1)

    with open(args.input_file, encoding="utf-8") as fh:
        lines = [x.strip() for x in fh.readlines() if "\t" in x]

    with Pool() as pool:
        results = pool.starmap(get_biterms_from_api, [(args, line) for line in lines])

    outfile = f"{args.output_file}.tsv"

    with open(outfile, mode="w", encoding="utf-8") as fh:
        for i, result in enumerate(results):
            if "error" in result:
                fh.write(f"line{i}\terror\t{result['error']}\n")
            else:
                for j, src_term in enumerate(result["src_terms"]):
                    fh.write(f"line{i}\t")
                    fh.write(f"{src_term}\t")
                    fh.write(f"{result['tgt_terms'][j]}\t")
                    fh.write(f"{result['similarities'][j]}\t")
                    fh.write(f"{result['frequencies'][j]}\t")
                    fh.write(f"{result['ranks'][j]}\n")
