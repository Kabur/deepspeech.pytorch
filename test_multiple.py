import argparse

import numpy as np
import torch
from tqdm import tqdm

from data.data_loader import SpectrogramDataset, AudioDataLoader
from decoder import GreedyDecoder
from opts import add_decoder_args, add_inference_args
from utils import load_model
import re
import os

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser = add_inference_args(parser)
parser.add_argument('--directory', required=True)

parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
parser.add_argument('--output-path', default=None, type=str, help="Where to save raw acoustic output")
parser = add_decoder_args(parser)
parser.add_argument('--save-output', action="store_true", help="Saves output of model from test")
args = parser.parse_args()

def get_last_number(str):
    numbers = re.findall(r'\d+', str)

    if numbers:
        return int(numbers[-1])
    else:
        return 0

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if args.cuda else "cpu")
    ratios = []
    model_paths = []

    for root, dirs, files in os.walk(args.directory):
        for file in files:
            model_paths.append(os.path.join(root, file))

    model_paths.sort(key=get_last_number)

    for model_path in model_paths:

        print("Testing:", model_path)
        try:
            model = load_model(device, model_path, args.cuda)
        except Exception as e:
            print("File " + model_path + " is probably not a model, skipping...")

        if args.decoder == "beam":
            from decoder import BeamCTCDecoder

            decoder = BeamCTCDecoder(model.labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                     cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                     beam_width=args.beam_width, num_processes=args.lm_workers)
        elif args.decoder == "greedy":
            decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
        else:
            decoder = None
        target_decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
        test_dataset = SpectrogramDataset(audio_conf=model.audio_conf, manifest_filepath=args.test_manifest,
                                          labels=model.labels, normalize=True)
        test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                      num_workers=args.num_workers)
        total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
        output_data = []
        non_empty = 0
        non_white = 0
        total = 0
        for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
            inputs, targets, input_percentages, target_sizes, filenames = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            inputs = inputs.to(device)
            # unflatten targets
            split_targets = []
            offset = 0
            for size in target_sizes:
                split_targets.append(targets[offset:offset + size])
                offset += size

            out, output_sizes = model(inputs, input_sizes)

            if args.save_output:
                # add output to data array, and continue
                output_data.append((out.cpu().numpy(), output_sizes.numpy()))

            decoded_output, _ = decoder.decode(out, output_sizes)
            target_strings = target_decoder.convert_to_strings(split_targets)
            for x in range(len(target_strings)):
                total += 1
                transcript, reference = decoded_output[x][0], target_strings[x][0]
                wer_inst = decoder.wer(transcript, reference)
                cer_inst = decoder.cer(transcript, reference)
                total_wer += wer_inst
                total_cer += cer_inst
                num_tokens += len(reference.split())
                num_chars += len(reference)
                if args.verbose:
                    print("Filename: ", filenames[x])
                    print("Ref:", reference.lower())
                    print("Hyp: \"" + transcript.lower() + "\"")
                    print("Hyp_code", [ord(c) for c in transcript.lower()])
                    print("WER:", float(wer_inst) / len(reference.split()), "CER:", float(cer_inst) / len(reference), "\n")
                if transcript.lower().strip() is not "":
                    non_white += 1
                if transcript.lower() is not "":
                    non_empty += 1


        wer = float(total_wer) / num_tokens
        cer = float(total_cer) / num_chars
        ratio = non_white / total
        ratios.append((model_path, ratio))

        for item in ratios:
            print("Model:", item[0].split("/")[-1])
            print("Ratio:", item[1])

        if args.save_output:
            np.save(args.output_path, output_data)

