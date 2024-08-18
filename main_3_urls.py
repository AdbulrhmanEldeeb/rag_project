import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.llms import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Load environment variables
load_dotenv()
mistral_image='mistral_image.png'
st.set_page_config('RAG',mistral_image)
# Load Hugging Face token
hf_token = os.getenv('HF_TOKEN')
st.sidebar.image(mistral_image)
def load_llm():
    """Load the language model."""
    repo_id = 'mistralai/Mistral-7B-Instruct-v0.2'
    llm = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=hf_token)
    return llm

# Load the LLM outside of the retrieval function to avoid reloading it multiple times
llm = load_llm()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
def initialize_chain(urls):
    """Initialize the retrieval chain with provided URLs."""
    try:
        # Load documents from the provided URLs
        loaders = UnstructuredURLLoader(urls=urls)
        data = loaders.load()

        # Ensure data is loaded correctly
        documents = []
        for i, content in enumerate(data):
            if content.page_content:
                documents.append(Document(page_content=content.page_content, metadata={"source": f"doc{i+1}"}))

        if not documents:
            st.error("Failed to load content from the provided URLs.")
            return None

        # Initialize the HuggingFace embedding model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        documents=text_splitter.split_documents(documents)
        # Create the FAISS index from the documents
        vector_index = FAISS.from_documents(documents, embeddings)

        # Create the RetrievalQAWithSourcesChain using the LLM and FAISS retriever
        chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=vector_index.as_retriever())
        return chain 
    except Exception as e:
        st.error(f"An error occurred during chain initialization: {str(e)}")
        return None

def retrieve_answer(query, chain):
    """Retrieve the answer using the provided query and chain."""
    try:
        result = chain({'question': query}, return_only_outputs=True)
        return result
    except Exception as e:
        st.error(f"An error occurred during retrieval: {str(e)}")
        return None

# Streamlit app layout
st.title('Retrieval-Augmented QA with LangChain')
st.sidebar.header('Input')

# Input fields for up to three URLs
query = st.text_input('Write your question here:')
url1 = st.sidebar.text_input('Add URL 1:')
url2 = st.sidebar.text_input('Add URL 2 (optional):')
url3 = st.sidebar.text_input('Add URL 3 (optional):')
add_urls = st.sidebar.button('Add URLs')

empty = st.empty()

# Initialize or retrieve the chain state
if 'chain' not in st.session_state:
    st.session_state.chain = None

if add_urls and (url1 or url2 or url3):
    urls = [url for url in [url1, url2, url3] if url]
    empty.text('Processing URLs, Wait a Second Please...')
    chain = initialize_chain(urls)
    if chain:
        st.session_state.chain = chain
        empty.text('Chain is ready. Add your question.')
    else:
        empty.text('Failed to initialize chain with the provided URLs.')

submit_query = st.button('Submit')

if submit_query and query and st.session_state.chain:
    result = retrieve_answer(query, st.session_state.chain)
    if result:
        answer = result.get('answer', 'No answer found.')
        source = result.get('sources', 'No source available.')
        st.write(f'**Answer:**\n{answer}')
        st.write(f'**Source:**\n{source}')
        
    else:
        st.write('No result to display.')
elif not query:
    st.info('Please enter a question.')
elif not st.session_state.chain:
    st.info('Please add at least one URL to initialize the chain.')
