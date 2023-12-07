import streamlit as st
import requests
import io
import PyPDF2
from utils.batch import split_text_into_batches, process_batches
from models.summarizer_model import load_summarizer_model
from utils.post_processing import clean_text

st.set_page_config(page_title="AI Summarizer Tool")

st.title("AI Summarizer Tool for Research Papers")

with st.spinner("Loading model... Please wait."):
    tokenizer, model = load_summarizer_model()

data_source = st.radio("Select data source:", ("URL", "File"))
text_input = ""

if data_source == "URL":
    url_input = st.text_input("Enter URL:")
    if url_input:
        try:
            response = requests.get(url_input)
            if response.headers.get("content-type") == "application/pdf":
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(response.content))
                text_input = "".join(
                    [
                        pdf_reader.pages[page_num].extract_text()
                        for page_num in range(len(pdf_reader.pages))
                    ]
                )
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching URL: {e}")

elif data_source == "File":
    file_input = st.file_uploader("Upload a file:", type="pdf")
    if file_input:
        if file_input.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(file_input)
            text_input = "".join(
                [
                    pdf_reader.pages[page_num].extract_text()
                    for page_num in range(len(pdf_reader.pages))
                ]
            )

button_placeholder = st.empty()
remaining_time_text = st.empty()

if button_placeholder.button("Summarize") and text_input:
    button_placeholder.empty()
    button_placeholder.divider()
    remaining_time_text.text("Estimated time remaining: -")

    loading_bar = st.progress(0)

    batches = split_text_into_batches(text_input)
    predictions = process_batches(
        batches, tokenizer, model, loading_bar, remaining_time_text
    )

    summary = [prediction.split("<n>")[0] for prediction in predictions]
    summary = "\n".join(summary)
    summary = clean_text(summary)

    remaining_time_text.empty()
    loading_bar.empty()

    st.subheader("Summary:")
    st.write(summary)
