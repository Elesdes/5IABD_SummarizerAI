from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer
import streamlit as st


@st.cache_resource()
def load_summarizer_model():
    summarizer_model = "google/bigbird-pegasus-large-arxiv"
    tokenizer = AutoTokenizer.from_pretrained(summarizer_model)
    model = BigBirdPegasusForConditionalGeneration.from_pretrained(
        summarizer_model, attention_type="original_full"
    )
    return tokenizer, model
