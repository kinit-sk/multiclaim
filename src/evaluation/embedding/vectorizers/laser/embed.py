#!/usr/bin/python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for document classification, bitext filtering
# and mining
#
# --------------------------------------------------------
#
# Tool to calculate to embed a text file
# The functions can be also imported into another Python code
#
# Source code: https://github.com/facebookresearch/LASER

import re
import os
import tempfile
import sys
import numpy as np
import logging
from collections import namedtuple
from subprocess import run
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from subprocess import run, CalledProcessError
import sentencepiece as spm

import torch
import torch.nn as nn


from fairseq.models.transformer import (
    Embedding,
    TransformerEncoder,
)
from fairseq.data.dictionary import Dictionary
from fairseq.modules import LayerNorm

SPACE_NORMALIZER = re.compile(r"\s+")
Batch = namedtuple("Batch", "srcs tokens lengths")

ROOT = './cache/laser'
ROMAN_LC = 'python3 src/evaluation/embedding/vectorizers/laser/romanize.py -l '
MOSES_BDIR = ROOT +'/tools-external/moses-tokenizer/'
REM_NON_PRINT_CHAR = MOSES_BDIR + 'remove-non-printing-char.perl'
NORM_PUNC = MOSES_BDIR + 'normalize-punctuation.perl -l '
DESCAPE = MOSES_BDIR + 'deescape-special-chars.perl'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def buffered_read(fp, buffer_size):
    buffer = []
    for src_str in fp:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer


def SPMApply(inp_fname, out_fname, spm_model, lang='en',
             descape=False,
             gzip=False):
    cat = 'zcat ' if gzip else 'cat '
    logger.info('SPM processing.')

    assert os.path.isfile(spm_model), f'SPM model {spm_model} not found'
    command = (cat + inp_fname
        + '|' + REM_NON_PRINT_CHAR
        + '|' + NORM_PUNC + lang
        + ('|' + DESCAPE if descape else '')
        + '|' + ROMAN_LC + 'none'
        + ' > ' + out_fname)
    try:
        run(["/bin/bash", "-o", "pipefail", "-c", command], check=True, capture_output=True)
        with open(out_fname, "r", encoding="utf-8", errors="surrogateescape") as fin:
            sentences = fin.readlines()
        sp = spm.SentencePieceProcessor(model_file=spm_model)
        encoded = []
        logger.info(f'Calculating encoding using {spm_model}.')
        for sentence in tqdm(sentences):
            encoded.append(' '.join(sp.encode(sentence.strip(), out_type=str)))
        return encoded
    except CalledProcessError as e:
        logger.error(e.stderr.decode().strip())
        sys.exit(1)



class SentenceEncoder:
    def __init__(
        self,
        model_path,
        max_sentences=None,
        max_tokens=None,
        spm_vocab=None,
        cpu=False,
        sort_kind="quicksort",
    ):
        logger.info(f"Loading encoder {model_path}.")
        self.use_cuda = torch.cuda.is_available() and not cpu
        self.max_sentences = max_sentences
        self.max_tokens = max_tokens
        if self.max_tokens is None and self.max_sentences is None:
            self.max_sentences = 1

        state_dict = torch.load(model_path)
        if "params" in state_dict:
            self.encoder = LaserLstmEncoder(**state_dict["params"])
            self.encoder.load_state_dict(state_dict["model"])
            self.dictionary = state_dict["dictionary"]
            self.prepend_bos = False
            self.left_padding = False
        else:
            self.encoder = LaserTransformerEncoder(state_dict, spm_vocab)
            self.dictionary = self.encoder.dictionary.indices
            self.prepend_bos = state_dict["cfg"]["model"].prepend_bos
            self.left_padding = state_dict["cfg"]["model"].left_pad_source
        del state_dict
        self.bos_index = self.dictionary["<s>"] = 0
        self.pad_index = self.dictionary["<pad>"] = 1
        self.eos_index = self.dictionary["</s>"] = 2
        self.unk_index = self.dictionary["<unk>"] = 3

        if self.use_cuda:
            logger.info('Transfer encoder to GPU.')
            self.encoder.cuda()
        self.encoder.eval()
        self.sort_kind = sort_kind

    def _process_batch(self, batch):
        tokens = batch.tokens
        lengths = batch.lengths
        if self.use_cuda:
            tokens = tokens.cuda()
            lengths = lengths.cuda()

        with torch.no_grad():
            sentemb = self.encoder(tokens, lengths)["sentemb"]
        embeddings = sentemb.detach().cpu().numpy()
        return embeddings

    def _tokenize(self, line):
        tokens = SPACE_NORMALIZER.sub(" ", line).strip().split()
        ntokens = len(tokens)
        if self.prepend_bos:
            ids = torch.LongTensor(ntokens + 2)
            ids[0] = self.bos_index
            for i, token in enumerate(tokens):
                ids[i + 1] = self.dictionary.get(token, self.unk_index)
            ids[ntokens + 1] = self.eos_index
        else:
            ids = torch.LongTensor(ntokens + 1)
            for i, token in enumerate(tokens):
                ids[i] = self.dictionary.get(token, self.unk_index)
            ids[ntokens] = self.eos_index
        return ids

    def _make_batches(self, lines):
        tokens = [self._tokenize(line) for line in lines]
        lengths = np.array([t.numel() for t in tokens])
        indices = np.argsort(-lengths, kind=self.sort_kind)

        def batch(tokens, lengths, indices):
            toks = tokens[0].new_full((len(tokens), tokens[0].shape[0]), self.pad_index)
            if not self.left_padding:
                for i in range(len(tokens)):
                    toks[i, : tokens[i].shape[0]] = tokens[i]
            else:
                for i in range(len(tokens)):
                    toks[i, -tokens[i].shape[0] :] = tokens[i]
            return (
                Batch(srcs=None, tokens=toks, lengths=torch.LongTensor(lengths)),
                indices,
            )

        batch_tokens, batch_lengths, batch_indices = [], [], []
        ntokens = nsentences = 0
        for i in indices:
            if nsentences > 0 and (
                (self.max_tokens is not None and ntokens + lengths[i] > self.max_tokens)
                or (self.max_sentences is not None and nsentences == self.max_sentences)
            ):
                yield batch(batch_tokens, batch_lengths, batch_indices)
                ntokens = nsentences = 0
                batch_tokens, batch_lengths, batch_indices = [], [], []
            batch_tokens.append(tokens[i])
            batch_lengths.append(lengths[i])
            batch_indices.append(i)
            ntokens += tokens[i].shape[0]
            nsentences += 1
        if nsentences > 0:
            yield batch(batch_tokens, batch_lengths, batch_indices)

    def encode_sentences(self, sentences):
        indices = []
        results = []
        for batch, batch_indices in self._make_batches(sentences):
            indices.extend(batch_indices)
            results.append(self._process_batch(batch))
        return np.vstack(results)[np.argsort(indices, kind=self.sort_kind)]


class LaserTransformerEncoder(TransformerEncoder):
    def __init__(self, state_dict, vocab_path):
        self.dictionary = Dictionary.load(vocab_path)
        if any(
            k in state_dict["model"]
            for k in ["encoder.layer_norm.weight", "layer_norm.weight"]
        ):
            self.dictionary.add_symbol("<mask>")
        cfg = state_dict["cfg"]["model"]
        self.sentemb_criterion = cfg.sentemb_criterion
        self.pad_idx = self.dictionary.pad_index
        self.bos_idx = self.dictionary.bos_index
        embed_tokens = Embedding(
            len(self.dictionary), cfg.encoder_embed_dim, self.pad_idx,
        )
        super().__init__(cfg, self.dictionary, embed_tokens)
        if "decoder.version" in state_dict["model"]:
            self._remove_decoder_layers(state_dict)
        if "layer_norm.weight" in state_dict["model"]:
            self.layer_norm = LayerNorm(cfg.encoder_embed_dim)
        self.load_state_dict(state_dict["model"])

    def _remove_decoder_layers(self, state_dict):
        for key in list(state_dict["model"].keys()):
            if not key.startswith(
                (
                    "encoder.layer_norm",
                    "encoder.layers",
                    "encoder.embed",
                    "encoder.version",
                )
            ):
                del state_dict["model"][key]
            else:
                renamed_key = key.replace("encoder.", "")
                state_dict["model"][renamed_key] = state_dict["model"].pop(key)

    def forward(self, src_tokens, src_lengths):
        encoder_out = super().forward(src_tokens, src_lengths)
        if isinstance(encoder_out, dict):
            x = encoder_out["encoder_out"][0]  # T x B x C
        else:
            x = encoder_out[0]
        if self.sentemb_criterion == "cls":
            cls_indices = src_tokens.eq(self.bos_idx).t()
            sentemb = x[cls_indices, :]
        else:
            padding_mask = src_tokens.eq(self.pad_idx).t().unsqueeze(-1)
            if padding_mask.any():
                x = x.float().masked_fill_(padding_mask, float("-inf")).type_as(x)
            sentemb = x.max(dim=0)[0]
        return {"sentemb": sentemb}


class LaserLstmEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings,
        padding_idx,
        embed_dim=320,
        hidden_size=512,
        num_layers=1,
        bidirectional=False,
        left_pad=True,
        padding_value=0.0,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.padding_idx = padding_idx
        self.embed_tokens = nn.Embedding(
            num_embeddings, embed_dim, padding_idx=self.padding_idx
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths):
        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.data.new(*state_size).zero_()
        c0 = x.data.new(*state_size).zero_()
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outs, padding_value=self.padding_value
        )
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:

            def combine_bidir(outs):
                return torch.cat(
                    [
                        torch.cat([outs[2 * i], outs[2 * i + 1]], dim=0).view(
                            1, bsz, self.output_units
                        )
                        for i in range(self.num_layers)
                    ],
                    dim=0,
                )

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        # Set padded outputs to -inf so they are not selected by max-pooling
        padding_mask = src_tokens.eq(self.padding_idx).t().unsqueeze(-1)
        if padding_mask.any():
            x = x.float().masked_fill_(padding_mask, float("-inf")).type_as(x)

        # Build the sentence embedding by max-pooling over the encoder outputs
        sentemb = x.max(dim=0)[0]

        return {
            "sentemb": sentemb,
            "encoder_out": (x, final_hiddens, final_cells),
            "encoder_padding_mask": encoder_padding_mask
            if encoder_padding_mask.any()
            else None,
        }


def load_model(
    encoder: str,
    spm_model: str,
    **encoder_kwargs
) -> SentenceEncoder:
    if spm_model:
        spm_vocab = str(Path(spm_model).with_suffix(".cvocab"))
    else:
        spm_vocab = None
    return SentenceEncoder(
        encoder, spm_vocab=spm_vocab, **encoder_kwargs
    )


def embed_sentences(
    sentences,
    batch_size,
    encoder_path: Path = None,
    spm_lang: Optional[str] = "en",
    spm_model: Optional[Path] = None,
    max_tokens: int = 12000,
    max_sentences: Optional[int] = None,
    cpu: bool = False,
    sort_kind: str = "quicksort",
):
    assert encoder_path, "Provide initialised encoder or encoder_path"
    batch_size = max(batch_size, 1)

    encoder = load_model(
        encoder_path,
        spm_model,
        max_sentences=max_sentences,
        max_tokens=max_tokens,
        sort_kind=sort_kind,
        cpu=cpu,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        if spm_model:
            spm_fname = os.path.join(tmpdir, "spm")
            out_fname = os.path.join(tmpdir, 'out')
            with open(spm_fname, 'w') as fp:
                for sentence in sentences:
                    sentence = sentence.replace("\n", " ")
                    fp.write(f'{sentence}\n')

            sentences = SPMApply(
                spm_fname,
                out_fname,
                spm_model,
                lang=spm_lang
            )

    result = None
    logger.info(f'Calculating embeddings using {encoder_path}.')
    for batch_sentences in tqdm(buffered_read(sentences, batch_size), total=len(sentences) // batch_size + 1):
        encoded = encoder.encode_sentences(batch_sentences)
        if result is None:
            result = encoded
        else:
            result = np.concatenate((result, encoded))
    
    return result
