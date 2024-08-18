# Retrieval-Augmented QA with LangChain

This repository contains a Streamlit application that leverages LangChain for a retrieval-augmented question-answering (QA) system. It allows users to input up to three URLs, processes the content from these URLs, and uses a language model to answer questions based on the retrieved information.

## Features

- **Document Retrieval:** Fetch and process documents from up to three provided URLs.
- **Text Splitting:** Split documents into chunks for more efficient processing.
- **Question Answering:** Use a language model to answer questions based on the retrieved content.
- **User Interface:** Simple and interactive UI built with Streamlit.

## Requirements

- Python 3.7+
- Streamlit
- LangChain
- dotenv
- HuggingFace API Token

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/repo-name.git
   cd repo-name
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Create a .env file in the root directory and add your HuggingFace API token:** 
  HF_TOKEN=your_huggingface_api_token

## Usage
1. **Run the Streamlit application:**
   ```bash
   streamlit run app.py

    
2. **Open the application in your browser, usually at http://localhost:8501.**

Use the app:
Input URLs:
Enter up to three URLs in the sidebar.
Submit Question: 
Type your question in the main input field and click "Submit" to retrieve an answer based on the content from the URLs.


