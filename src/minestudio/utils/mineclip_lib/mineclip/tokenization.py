from __future__ import annotations
import os
from functools import lru_cache

# disable HuggingFace warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModel


def get_model_full_name(model_name):
    if model_name in ["bert", "bert-base-uncased"]:
        model_full_name = "bert-base-uncased"
    elif model_name in ["distilbert", "distilbert-base-uncased"]:
        model_full_name = "distilbert-base-uncased"
    elif model_name in ["clip", "openai/clip-vit-base-patch16"]:
        model_full_name = "openai/clip-vit-base-patch16"
    else:
        model_full_name = model_name
    return model_full_name


@lru_cache
def get_tokenizer(model_name, use_fast: bool = True):
    """
    It is LRU-cached because it is slow to load tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        get_model_full_name(model_name), use_fast=use_fast
    )
    return tokenizer


def tokenize_batch(
    texts: str | list[str] | list[list[int]], max_length: int, language_model="clip"
):
    """
    Args:
        texts: str or [str] or [[int ids], [int...]], len(text) = batch_size

    Returns:
        torch.LongTensor of shape [batch_size, max_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    if language_model == "clip":
        assert (
            max_length <= 77
        ), "clip language model supports max length of 77 (including BOS and EOS tokens)"
        max_length = 77

    tokenizer = get_tokenizer(language_model)
    # text_tokens: list[list[int]]
    if isinstance(texts[0], list) and isinstance(texts[0][0], int):
        text_tokens = texts
    else:
        text_tokens = tokenizer(texts, add_special_tokens=False)["input_ids"]
    if language_model == "clip":
        begin_token_id = tokenizer.bos_token_id
        # CLIP padding tokens are 0 for openAI repo code to work
        pad_token_id = 0
        end_token_id = tokenizer.eos_token_id
    elif language_model == "bert":
        begin_token_id = tokenizer.cls_token_id
        pad_token_id = tokenizer.pad_token_id
        end_token_id = tokenizer.sep_token_id
    else:
        raise NotImplementedError

    batch_tokens = torch.ones((len(texts), max_length), dtype=torch.long) * pad_token_id
    batch_tokens[:, 0] = begin_token_id
    for i, sentence in enumerate(text_tokens):
        sentence = sentence[: max_length - 2]
        batch_tokens[i, 1 : len(sentence) + 1] = torch.LongTensor(sentence)
        batch_tokens[i, len(sentence) + 1] = end_token_id
    return batch_tokens
