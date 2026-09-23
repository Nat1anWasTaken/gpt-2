from hashlib import sha256

from main import (
    gpt2_logits,
    load_embedding_layers,
    load_final_layer_norm,
    load_weights,
    load_tokenizer,
)
from datasets import load_dataset
import torch

def get_prompt(subject, e1, e2, e3, e4, row):
    letters = "ABCD"
    return f"""The following are multiple choice questions about {subject}.

{e1['question']}
(A) {e1['choices'][0]} (B) {e1['choices'][1]} (C) {e1['choices'][2]} (D) {e1['choices'][3]}
Answer: {letters[e1['answer']]}

{e2['question']}
(A) {e2['choices'][0]} (B) {e2['choices'][1]} (C) {e2['choices'][2]} (D) {e2['choices'][3]}
Answer: {letters[e2['answer']]}

{e3['question']}
(A) {e3['choices'][0]} (B) {e3['choices'][1]} (C) {e3['choices'][2]} (D) {e3['choices'][3]}
Answer: {letters[e3['answer']]}

{e4['question']}
(A) {e4['choices'][0]} (B) {e4['choices'][1]} (C) {e4['choices'][2]} (D) {e4['choices'][3]}
Answer: {letters[e4['answer']]}

{row['question']}
(A) {row['choices'][0]} (B) {row['choices'][1]} (C) {row['choices'][2]} (D) {row['choices'][3]}
Answer: """

def prepare_gpt2():
    """Load and return the shared GPT-2 components needed for inference."""
    tokenizer = load_tokenizer()
    weights = load_weights()
    wte, wpe = load_embedding_layers(weights)
    final_ln = load_final_layer_norm(weights)

    return tokenizer, weights, wte, wpe, final_ln


def mmlu_eval() -> dict[str, str]:
    dataset = load_dataset("cais/mmlu", "all", split="test")
    dev = load_dataset("cais/mmlu", "all", split="dev")
    tokenizer, weights, wte, wpe, final_ln = prepare_gpt2()
    predictions = {}

    for row in dataset.select(range(10)):
        e1, e2, e3, e4 = [example for example in dev if example["subject"] == row["subject"]][:4]
        subject = row["subject"].replace("_", " ")
        prompt = get_prompt(subject, e1, e2, e3, e4, row)

        token_ids = tokenizer.encode(prompt, add_special_tokens=False).ids

        with torch.inference_mode():
            logits = gpt2_logits(token_ids, weights, wte, wpe, final_ln)

        answer_token_ids = [
            tokenizer.encode(f" {letter}", add_special_tokens=False).ids[0]
            for letter in "ABCD"
        ]

        choice_logits = logits[answer_token_ids]
        question_hash = sha256(row["question"].encode("utf-8")).hexdigest()
        predictions[question_hash] = "ABCD"[int(choice_logits.argmax())]

    return predictions


if __name__ == "__main__":
    print(mmlu_eval())
