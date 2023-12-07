import nltk
import streamlit as st
import time


def split_text_into_batches(text, batch_size=4096):
    sentences = nltk.sent_tokenize(text)

    current_batch = []
    current_batch_size = 0
    batches = []

    for sentence in sentences:
        if current_batch_size + len(sentence) <= batch_size:
            current_batch.append(sentence)
            current_batch_size += len(sentence)
        else:
            batches.append(" ".join(current_batch))
            current_batch = [sentence]
            current_batch_size = len(sentence)

    if current_batch:
        batches.append(" ".join(current_batch))

    return batches


def process_batches(batches, tokenizer, model, progress_bar, remaining_time_text):
    predictions = []

    start_time = time.time()

    with st.spinner("Summarizing... Please wait."):
        for i, batch in enumerate(batches):
            inputs = tokenizer(batch, return_tensors="pt")  # Max length = 4096
            prediction = model.generate(**inputs)
            prediction = tokenizer.batch_decode(prediction)
            predictions.extend(prediction)

            progress_bar.progress((i + 1) / len(batches))

            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / (i + 1) * len(batches)
            remaining_time = max(estimated_total_time - elapsed_time, 0)

            minutes, seconds = divmod(int(remaining_time), 60)
            remaining_time_text.text(
                f"Estimated time remaining: {minutes:02d}:{seconds:02d} minutes."
            )

    return predictions
