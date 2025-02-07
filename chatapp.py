import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
import os

st.title("📝 File Q&A with HuggingFace")

# Get the HuggingFace API key from environment variable
hf_api_key = os.getenv("HF_TOKEN")

# Sidebar to display the GitHub info
with st.sidebar:
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/1_File_Q%26A.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

# File upload and question input
uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"))
question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

# Check if the API key is missing
if uploaded_file and question and not hf_api_key:
    st.info("Please add your HuggingFace API key to the environment variables to continue.")

# If both file and question are provided and API key is available
if uploaded_file and question and hf_api_key:
    article = uploaded_file.read().decode()

    # Set up the HuggingFace model inference using HuggingFaceEndpoint
    model_id = "gpt2"  # You can replace this with your desired HuggingFace model (e.g., GPT-3, GPT-Neo, etc.)
    hf = HuggingFaceEndpoint(
        repo_id=model_id,
        task="text-generation",  # Specifying the task type as text generation
        max_new_tokens=100,
        temperature=0.7,
        token=hf_api_key,  # Use the API key from the environment variable
    )

    # Prepare the prompt for the question
    prompt = f"Here is an article:\n\n{article}\n\nQuestion: {question}\nAnswer:"

    # Get the response from the model
    response = hf(prompt)

    # Display the response
    st.write("### Answer")
    st.write(response['generated_text'])