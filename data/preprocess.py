import argparse
import os
from dictionary import Dictionary
import tokenizer
import shutil


def get_preprocessing_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-s", "--source_lang", default=None, metavar="SRC",
						help="source language")
	parser.add_argument("-t", "--target_lang", default=None, metavar="TARGET",
						help="target language")
	parser.add_argument("--trainpref", metavar="FP", default=None,
						help="train file prefix (also used to build dictionaries)")
	parser.add_argument("--validpref", metavar="FP", default=None,
						help="comma separated, valid file prefixes ")
	parser.add_argument("--testpref", metavar="FP", default=None,
						help="comma separated, test file prefixes ")
	parser.add_argument("--destdir", metavar="DIR", default="data-bin",
						help="destination dir")
	return parser


def build_dictionary(filename, threshold=-1, nwords=-1):
	d = Dictionary()
	Dictionary.add_file_to_dictionary(filename, d, tokenizer.tokenize_line)
	d.finalize(threshold=threshold, nwords=nwords)
	return d


def make_dataset(input_prefix, output_prefix, src_lang, tgt_lang):
	input_src_file = f"{input_prefix}.{src_lang}"
	output_src_file = f"{output_prefix}.{src_lang}"
	input_tgt_file = f"{input_prefix}.{tgt_lang}"
	output_tgt_file = f"{output_prefix}.{tgt_lang}"
	shutil.copyfile(input_src_file, output_src_file)
	shutil.copyfile(input_tgt_file, output_tgt_file)


def main(args):
	os.makedirs(args.destdir, exist_ok=True)
	src_dict = build_dictionary(f"{args.trainpref}.{args.source_lang}")
	tgt_dict = build_dictionary(f"{args.trainpref}.{args.target_lang}")
	src_dict.save(os.path.join(args.destdir, f"dict.{args.source_lang}.txt"))
	tgt_dict.save(os.path.join(args.destdir, f"dict.{args.target_lang}.txt"))
	make_dataset(args.trainpref, f"{args.destdir}/train", args.source_lang, args.target_lang)
	if args.validpref:
		make_dataset(args.validpref, f"{args.destdir}/valid", args.source_lang, args.target_lang)
	if args.testpref:
		make_dataset(args.testpref, f"{args.destdir}/test", args.source_lang, args.target_lang)


if __name__ == "__main__":
	parser = get_preprocessing_parser()
	args = parser.parse_args()
	main(args)
