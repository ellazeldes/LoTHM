#!/usr/bin/env python3
# Copyright    2023                            (authors: Feiteng Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Phonemize Text and EnCodec Audio.

Usage example:
    python3 bin/infer.py \
        --decoder-dim 128 --nhead 4 --num-decoder-layers 4 --model-name valle \
        --text-prompts "Go to her." \
        --audio-prompts ./prompts/61_70970_000007_000001.wav \
        --output-dir infer/demo_valle_epoch20 \
        --checkpoint exp/valle_nano_v2/epoch-20.pt

"""
import argparse
import logging
import os
import sys
from pathlib import Path
import json
import re


os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


import torch
import time
from utils import AttributeDict

from tokenizer import (
    AudioTokenizer,
    tokenize_audio,
)
from valle.data.collation import get_text_token_collater
from valle.models import get_model
from valle.data.hebrew_root_tokenizer import AlefBERTRootTokenizer


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text-prompts",
        type=str,
        default="",
        help="Text prompt (transcription of audio prompt).",
    )

    parser.add_argument(
        "--audio-prompts",
        type=str,
        default="",
        help="Audio prompts",
    )

    parser.add_argument(
        "--text",
        type=str,
        default="sentences.txt",
        help="Text to be synthesized.",
    )

    parser.add_argument(
        "--lm-checkpoint",
        type=str,
        help="Path to the saved checkpoint of LM.",
    )

    parser.add_argument(
        "--vocoder-checkpoint",
        type=str,
        help="Path to the saved checkpoint of Vocoder.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./out"),
        help="Path to the tokenized files.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=-100,
        help="Whether AR Decoder do top_k(if > 0) sampling.",
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature of AR Decoder top_k sampling.",
    )
    
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="The window size for sampling with threshold ratio.",
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="The top p value for sampling.",
    )
    
    parser.add_argument(
        "--repetition-threshold",
        type=float,
        default=0.2,
        help="The threshold for sampling.",
    )

    return parser.parse_args()


def load_model(checkpoint, device):
    if not checkpoint:
        return None

    checkpoint = torch.load(checkpoint, map_location=device)

    args = AttributeDict(checkpoint)
    model = get_model(args)

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model"], strict=True
    )
    assert not missing_keys
    model.to(device)
    model.eval()

    text_tokens = args.text_tokens

    return model, text_tokens

def remove_niqqud_from_string(my_string):
    return ''.join(['' if  1456 <= ord(c) <= 1479 else c for c in my_string])

def has_english_chars(input_string):
            pattern = re.compile(r'[A-Za-z0-9]')
            # Search the string for the pattern
            return bool(pattern.search(input_string))

def remove_paranthesis(input_string):
    # Regular expression to match parentheses
    pattern = re.compile(r'[()]')
    
    # Substitute the matched patterns (parentheses) with an empty string
    return pattern.sub('', input_string)
            
@torch.no_grad()
def main():
    args = get_args()
    print(args)
    text_tokenizer = AlefBERTRootTokenizer(vocab_file=f"{os.path.dirname(os.path.abspath(__file__))}/tokenizer/vocab.txt")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    model, text_tokens = load_model(args.lm_checkpoint, device)
    text_tokens = f"{os.path.dirname(os.path.abspath(__file__))}/tokenizer/unique_words_tokens_all.k2symbols"
    text_collater = get_text_token_collater(text_tokens)

    print("using hubert to encode audio prompt")
    audio_tokenizer = AudioTokenizer()
   
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if isinstance(args.text, str):
        text_lines = args.text.split("|")
    else:
        with open(args.text) as f:
            text_lines = f.readlines()

    data = []
    for idx, line in enumerate(text_lines):
        line = remove_niqqud_from_string(line)
        if has_english_chars(line):
            continue
        logging.info(f"synthesize text: {line}")
        text_tokens, text_tokens_lens = text_collater(
            [
                text_tokenizer._tokenize(
                    f"{args.text_prompts} {line}".strip().replace(" ", "_")
                )
            ]
        )
        _, enroll_x_lens = text_collater(
            [
                text_tokenizer._tokenize(
                    f"{args.text_prompts}".strip()
                )
            ]
        )

        audio_prompts = tokenize_audio(audio_tokenizer, args.audio_prompts)
        audio_prompts = audio_prompts.to(device)
        # synthesis
        encoded_frames = model.inference(
            text_tokens.to(device),
            text_tokens_lens.to(device),
            audio_prompts.unsqueeze(0).unsqueeze(-1),
            enroll_x_lens=enroll_x_lens,
            window_size=args.window_size,
            top_p_value=args.top_p,
            repetition_threshold=args.repetition_threshold,
            top_k=args.top_k,
            temperature=args.temperature,
            device=device
        )

        l =list(encoded_frames.squeeze().cpu().numpy())
        data.append({
            'audio': args.audio_prompts,
            'hubert': " ".join([str(c) for c in l])
        })
    
    tmp_file = os.path.join(args.output_dir, "tmp_input_code.txt")
    with open(tmp_file, "w") as f:
        for dictionary in data:
        # Write each dictionary on a new line
            f.write(f"{dictionary}\n")

                
    
    audio_tokenizer.decode(
        tmp_file, 
        output_dir=args.output_dir,
        device=device,
        checkpoint=args.vocoder_checkpoint
    )
            
        

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)
if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
