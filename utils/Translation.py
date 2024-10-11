import requests
import sys
import io
import os
import re
import numpy as np

from pathlib import Path

import pandas as pd
import torch

# import evaluate

from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from tqdm.notebook import tqdm

from tokenizers import Tokenizer
from tokenizers import normalizers
from tokenizers import pre_tokenizers
from tokenizers.models import WordLevel, WordPiece
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import NFC, StripAccents, Strip, Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer, WordPieceTrainer

from sklearn.model_selection import train_test_split


def translate_openl(txt):
    url = "https://openl-translate.p.rapidapi.com/translate"

    payload = {"target_lang": "arz", "text": txt}
    headers = {
        "x-rapidapi-key": "394717c44cmsh82600cc7cdcfb18p1008a8jsne46053c531d3",
        "x-rapidapi-host": "openl-translate.p.rapidapi.com",
        "Content-Type": "application/json",
    }

    response = requests.post(url, json=payload, headers=headers)
    return response.json()["translatedText"]


class Variables:
    PATHS = {
        "Data": "D:\\DEPI\\Esma3nyAPI\\ara_eng.txt",
        # "HuggingFace": "zeyadusf/translation-EnAr",
    }
    TEXT = {
        "seq_len": 20,
        "batch": 32,
    }
    TRANSFORMER = {
        "heads": 8,
        "latent_dim": 512,
        "pf_dim": 2048,
        "dropout": 0.25,
        "encoder_layers": 3,
        "decoder_layers": 3,
    }
    TRAIN = {
        "epoch": 120,
        "lr": 1e-4,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


var = Variables()
punc_re = re.compile(r"""[!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~،؟…«“\":\"…”]""")


def remove_punctation(text: str) -> str:
    return punc_re.sub(repl="", string=text)


diactircs_re = re.compile("[\u064B-\u0652]")


def remove_diactrics(text: str) -> str:
    return diactircs_re.sub(repl="", string=text)


def clean_text(text: str) -> str:
    return remove_punctation(remove_diactrics(text))


class TranslationDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_values(by="length")
        self.src = df["source"].tolist()
        self.trg = df["target"].tolist()

    def __getitem__(self, idx):
        source = self.src[idx]
        target = self.trg[idx]
        return (source, target)

    def __len__(self):
        return len(self.df)


def prepare_batch(batch, ar_tokenizer: Tokenizer, en_tokenizer: Tokenizer):
    ar = list(map(lambda x: x[0], batch))
    en = list(map(lambda x: x[1], batch))
    ar = ar_tokenizer.encode_batch(ar)
    en = en_tokenizer.encode_batch(en)
    ar_tokens = list(map(lambda x: x.ids, ar))
    ar_mask = list(map(lambda x: x.attention_mask, ar))
    en_tokens = list(map(lambda x: x.ids, en))
    en_mask = list(map(lambda x: x.attention_mask, en))
    source = torch.tensor(ar_tokens)
    source_mask = torch.tensor(ar_mask)
    target = torch.tensor(en_tokens)
    target_mask = torch.tensor(en_mask)
    return (source, source_mask, target, target_mask)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MultiHeadAttention(nn.Module):
    def __init__(self, latent_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_heads = n_heads
        self.head_dim = self.latent_dim // self.n_heads
        self.fc_q = nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim)
        self.fc_k = nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim)
        self.fc_v = nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim)
        self.fc_o = nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(query)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = self.dropout(torch.softmax(energy, dim=-1))
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.latent_dim)
        x = self.fc_o(x)
        return x, attention


class PositionWiseFeedForward(nn.Module):
    def __init__(self, latent_dim: int, pf_dim: int, dropout: float):
        super().__init__()

        self.fc1 = nn.Linear(in_features=latent_dim, out_features=pf_dim)
        self.fc2 = nn.Linear(in_features=pf_dim, out_features=latent_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, latent_dim: int, n_heads: int, pf_dim: int, dropout: float):
        super().__init__()

        self.attention = MultiHeadAttention(
            latent_dim=latent_dim, n_heads=n_heads, dropout=dropout
        )
        self.positionwise_feedforward = PositionWiseFeedForward(
            latent_dim=latent_dim, pf_dim=pf_dim, dropout=dropout
        )

        self.attn_norm = nn.LayerNorm(normalized_shape=latent_dim)
        self.ppff_norm = nn.LayerNorm(normalized_shape=latent_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src, src_mask):
        # src = [batch_size, src_len, latent_dim]
        # src_mask = [batch_size, 1, 1, src_len]

        attention_values, _ = self.attention(src, src, src, src_mask)
        # attention_values = [batch_size, src_len, latent_dim]

        output = attention_values + src
        # output = [batch_size, src_len, latent_dim]

        ppff = self.positionwise_feedforward(output)
        # ppff = [batch_size, src_len, latent_dim]

        output = self.ppff_norm(output + self.dropout(ppff))
        # output = [batch_size, src_len, latent_dim]
        return output


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        latent_dim: int,
        n_layers: int,
        n_heads: int,
        pf_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.tok_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=latent_dim
        )
        self.pos_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=latent_dim
        )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    latent_dim=latent_dim,
                    n_heads=n_heads,
                    pf_dim=pf_dim,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src, src_mask):
        # src = [batch_size, src_len]
        # src_mask = [batch_size, 1, 1, src_len]

        batch_size, src_len = src.shape

        positions = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(src)
        # positions = [batch_size, src_len]

        scale = torch.sqrt(torch.FloatTensor([self.tok_embedding.embedding_dim])).to(
            src
        )
        token_embeddings = self.tok_embedding(src) * scale
        positional_embeddings = self.pos_embedding(positions)
        # token_embeddings = [batch_size, src_len, latent_dim]
        # positional_embeddings = [batch_size, src_len, latent_dim]

        output = self.dropout(token_embeddings + positional_embeddings)
        # embeddings = [batch_size, src_len, latent_dim]

        for layer in self.layers:
            output = layer(output, src_mask)
        # output = [batch_size, src_len, latent_dim]

        return output


class DecoderLayer(nn.Module):
    def __init__(self, latent_dim: int, n_heads: int, pf_dim: int, dropout: float):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            latent_dim=latent_dim, n_heads=n_heads, dropout=dropout
        )
        self.encoder_attention = MultiHeadAttention(
            latent_dim=latent_dim, n_heads=n_heads, dropout=dropout
        )
        self.positionwise_feedfoward = PositionWiseFeedForward(
            latent_dim=latent_dim, pf_dim=pf_dim, dropout=dropout
        )

        self.self_attn_norm = nn.LayerNorm(normalized_shape=latent_dim)
        self.enc_attn_norm = nn.LayerNorm(normalized_shape=latent_dim)
        self.pff_norm = nn.LayerNorm(normalized_shape=latent_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, trg, enc_out, trg_mask, src_mask):
        # trg = [batch_size, trg_len, latent_dim]
        # enc_src = [batch_size, src_len, latent_dim]
        # trg_mask = [batch_size, 1, trg_len, trg_len]
        # src_mask = [batch_size, 1, 1, src_len]

        self_attention, _ = self.self_attention(trg, trg, trg, trg_mask)

        output = self.self_attn_norm(trg + self.dropout(self_attention))
        # output = [batch_size, trg_len, latent_dim]

        encoder_attention, attention_scores = self.encoder_attention(
            output, enc_out, enc_out, src_mask
        )
        # encoder_attention = [batch_size, trg_len, latent_dim]

        output = self.enc_attn_norm(output + self.dropout(encoder_attention))
        # output = [batch_size, trg_len, latent_dim]

        pwff = self.positionwise_feedfoward(output)
        output = self.pff_norm(output + self.dropout(pwff))

        return output, attention_scores


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        latent_dim: int,
        n_layers: int,
        n_heads: int,
        pf_dim: int,
        dropout: float,
        max_len: int = 100,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.tok_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=latent_dim
        )
        self.pos_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=latent_dim
        )

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    latent_dim=latent_dim,
                    n_heads=n_heads,
                    pf_dim=pf_dim,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.fc_out = nn.Linear(in_features=latent_dim, out_features=vocab_size)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, trg, enc_out, trg_mask, src_mask):
        # trg = [batch_size, trg_len]
        # enc_out = [batch_size, src_len, latent_dim]
        # trg_mask = [batch_size, 1, trg_len, trg_len]
        # src_mask = [batch_size, 1, 1, src_len]

        batch_size, trg_len = trg.shape

        positions = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(trg)
        # positions = [batch_size, trg_len]

        scale = torch.sqrt(torch.FloatTensor([self.latent_dim])).to(trg)
        output = (self.tok_embedding(trg) * scale) + self.pos_embedding(positions)
        output = self.dropout(output)
        # output = [batch_size, trg_len, latent_dim]

        for layer in self.layers:
            output, attention = layer(output, enc_out, trg_mask, src_mask)
        # output = [batch_size, trg_len, latent_dim]

        output = self.dropout(output)
        output = self.fc_out(output)

        # output = [batch_size, trg_len, vocab_size]
        return output, attention


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def make_trg_mask(self, trg_pad_mask):
        # trg_pad_mask = [batch_size, trg_len]
        trg_len = trg_pad_mask.shape[1]

        trg_pad_mask = trg_pad_mask.unsqueeze(1).unsqueeze(2)

        trg_sub_mask = (
            torch.tril(torch.ones((trg_len, trg_len))).to(trg_pad_mask).bool()
        )

        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask

    def make_src_mask(self, src_pad_mask):
        return src_pad_mask.unsqueeze(1).unsqueeze(2)

    def forward(self, src, trg, src_mask, trg_mask):
        # src = [batch_size, src_len]
        # trg = [batch_size, trg_len]
        # src_mask = [batch_size, src_len]
        # trg_mask = [batch_size, trg_len]

        src_mask = self.make_src_mask(src_pad_mask=src_mask)
        trg_mask = self.make_trg_mask(trg_pad_mask=trg_mask)

        encoder_outputs = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, encoder_outputs, trg_mask, src_mask)
        # output [batch_size, trg_len, vocab_size]
        # attention = [batch_size, n_heads, trg_len, src_len]

        return output, attention


def translate_sentence(
    sentence: str,
    src_tokenizer: Tokenizer,
    trg_tokenizer: Tokenizer,
    model: nn.Module,
    device: torch.device,
    max_len: int = 50,
):

    model.eval()

    # encode sentence
    src = src_tokenizer.encode(sentence)
    # get src input as ids and attention_mask
    src_input = torch.tensor(src.ids).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(
        torch.tensor(src.attention_mask).unsqueeze(0).to(device)
    )

    with torch.no_grad():
        enc_src = model.encoder(src_input, src_mask)

    trg_indices = [trg_tokenizer.token_to_id("[SOS]")]

    for i in range(max_len):
        trg_inputs = torch.tensor(trg_indices).unsqueeze(0).to(device)

        trg_mask = [
            1 if token != trg_tokenizer.token_to_id("[PAD]") else 0
            for token in trg_indices
        ]
        trg_mask = torch.tensor(trg_mask).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_mask)

        output, attention = model.decoder(trg_inputs, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indices.append(pred_token)

        if pred_token == trg_tokenizer.token_to_id("[EOS]"):
            break

    return trg_tokenizer.decode(trg_indices), attention


def train_and_save_tokenizers():
    data = pd.read_csv(var.PATHS["Data"], sep="\t", names=["target", "source"])
    data["target"] = data["target"].map(clean_text)
    data["source"] = data["source"].map(clean_text)
    data = data.copy(deep=True)[~data.duplicated()]
    data["length"] = data["source"].map(lambda x: len(x.split(" ")))
    data = data[data["length"] <= var.TEXT["seq_len"]]

    ar_tokenizer = Tokenizer(model=WordLevel(unk_token="[OOV]"))
    en_tokenizer = Tokenizer(model=WordLevel(unk_token="[OOV]"))

    normalizer = normalizers.Sequence([NFC(), StripAccents(), Lowercase(), Strip()])
    pre_tokenizer = pre_tokenizers.Sequence([Whitespace()])
    post_processor = TemplateProcessing(
        single="[SOS] $A [EOS]", special_tokens=[("[SOS]", 2), ("[EOS]", 3)]
    )

    ar_tokenizer.normalizer = normalizer
    en_tokenizer.normalizer = normalizer
    ar_tokenizer.pre_tokenizer = pre_tokenizer
    en_tokenizer.pre_tokenizer = pre_tokenizer

    ar_tokenizer.enable_padding(direction="right")
    en_tokenizer.enable_padding(direction="right")
    ar_tokenizer.post_processor = post_processor
    en_tokenizer.post_processor = post_processor

    trainer = WordLevelTrainer(
        vocab_size=200000,
        show_progress=False,
        min_frequency=5,
        special_tokens=["[PAD]", "[OOV]", "[SOS]", "[EOS]"],
    )

    ar_tokenizer.train_from_iterator(iterator=data["source"].tolist(), trainer=trainer)
    en_tokenizer.train_from_iterator(iterator=data["target"].tolist(), trainer=trainer)

    # Save tokenizers
    ar_tokenizer.save("ar_tokenizer.json")
    en_tokenizer.save("en_tokenizer.json")

    return ar_tokenizer, en_tokenizer


def translate(sentence):
    ar_tokenizer = Tokenizer.from_file("ar_tokenizer.json")
    en_tokenizer = Tokenizer.from_file("en_tokenizer.json")

    encoder = Encoder(
        vocab_size=ar_tokenizer.get_vocab_size(),
        latent_dim=var.TRANSFORMER["latent_dim"],
        n_layers=var.TRANSFORMER["encoder_layers"],
        n_heads=var.TRANSFORMER["heads"],
        pf_dim=var.TRANSFORMER["pf_dim"],
        dropout=var.TRANSFORMER["dropout"],
    )

    decoder = Decoder(
        vocab_size=en_tokenizer.get_vocab_size(),
        latent_dim=var.TRANSFORMER["latent_dim"],
        n_layers=var.TRANSFORMER["decoder_layers"],
        n_heads=var.TRANSFORMER["heads"],
        pf_dim=var.TRANSFORMER["pf_dim"],
        dropout=var.TRANSFORMER["dropout"],
    )

    model = Seq2Seq(encoder=encoder, decoder=decoder).to(var.device)
    model_path = "translate_v1.pth"
    model.load_state_dict(torch.load(model_path, weights_only=True))
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(var.device)
    model.eval()

    translation, atn = translate_sentence(
        sentence=sentence,
        src_tokenizer=ar_tokenizer,
        trg_tokenizer=en_tokenizer,
        model=model,
        device=var.device,
        max_len=20,
    )
    # print(translation)
    return translation


# print(translate("سوف نذهب الي الحديقة الاحد المقبل"))
