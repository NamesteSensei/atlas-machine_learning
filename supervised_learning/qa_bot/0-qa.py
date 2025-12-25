#!/usr/bin/env python3
"""
Task 0: Extractive question answering using BERT.

The function `question_answer` uses:
- TensorFlow Hub model: bert-uncased-tf2-qa
- HuggingFace tokenizer: bert-large-uncased-whole-word-masking-finetuned-squad
"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Finds an answer to `question` inside `reference`.

    This is an extractive QA approach: the answer is a span of text from
    the reference document.

    Args:
        question (str): The question to answer.
        reference (str): The reference text to search for the answer.

    Returns:
        str or None: The extracted answer or None if no valid answer is found.
    """
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    encoded = tokenizer(
        question,
        reference,
        return_tensors="tf",
        truncation=True
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    token_type_ids = encoded["token_type_ids"]

    start_logits, end_logits = model(
        [input_ids, attention_mask, token_type_ids]
    )

    start_index = int(tf.argmax(start_logits, axis=1).numpy()[0])
    end_index = int(tf.argmax(end_logits, axis=1).numpy()[0]) + 1

    if start_index >= end_index:
        return None

    answer_ids = input_ids[0][start_index:end_index]
    answer_tokens = tokenizer.convert_ids_to_tokens(answer_ids)
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    answer = answer.strip()
    if not answer:
        return None

    return answer
