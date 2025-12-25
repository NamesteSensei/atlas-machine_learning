#!/usr/bin/env python3
"""
Task 0: Extractive question answering using BERT.
"""

from transformers import BertTokenizer, TFBertForQuestionAnswering
import tensorflow as tf


def question_answer(question, reference):
    """
    Answers a question based on a reference document using BERT.

    Args:
        question (str): Question to answer.
        reference (str): Reference text.

    Returns:
        str or None: Extracted answer or None if no answer found.
    """
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )
    model = TFBertForQuestionAnswering.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )

    inputs = tokenizer(
        question,
        reference,
        return_tensors="tf",
        truncation=True
    )

    input_ids = inputs["input_ids"]
    outputs = model(inputs)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_index = int(tf.argmax(start_logits, axis=1).numpy()[0])
    end_index = int(tf.argmax(end_logits, axis=1).numpy()[0]) + 1

    if start_index >= end_index:
        return None

    answer_ids = input_ids[0][start_index:end_index]
    answer = tokenizer.decode(answer_ids)

    if not answer.strip() or answer in ("[CLS]", "[SEP]"):
        return None

    return answer.strip()
