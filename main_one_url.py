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

# Load Hugging Face token
hf_token = os.getenv('HF_TOKEN')

def load_llm():
    """Load the language model."""
    repo_id = 'mistralai/Mistral-7B-Instruct-v0.2'
    llm = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=hf_token)
    return llm

# Load the LLM outside of the retrieval function to avoid reloading it multiple times
llm = load_llm()

text_splitter = RecursiveCharacterTextSplitter(
    separators = ["\n\n", "\n", " "],  # List of separators based on requirement (defaults to ["\n\n", "\n", " "])
    chunk_size = 200,  # size of each chunk created
    chunk_overlap  = 0,  # size of  overlap between chunks in order to maintain the context
    length_function = len  # Function to calculate size, currently we are using "len" which denotes length of string however you can pass any token counter)
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

        # Create the FAISS index from the documents
        vector_index = FAISS.from_documents(documents, embeddings)
        documents=text_splitter.split_documents(documents)

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

# Input fields
query = st.text_input('Write your question here:')
url = st.sidebar.text_input('Add URL that you want to retrieve data from:')
add_url = st.sidebar.button('Add URL')
empty = st.empty()

# Initialize or retrieve the chain state
if 'chain' not in st.session_state:
    st.session_state.chain = None

if add_url and url:
    empty.text('Processing URL...')
    chain = initialize_chain([url])
    if chain:
        st.session_state.chain = chain
        empty.text('Chain is ready. Add your question.')
    else:
        empty.text('Failed to initialize chain with the provided URL.')

submit_query = st.button('Submit')

if submit_query and query and st.session_state.chain:
    result = retrieve_answer(query, st.session_state.chain)
    if result:
        answer = result.get('answer', 'No answer found.')
        source = result.get('sources', 'No source available.')
        st.write(f'**Answer:**\n{answer}')
        st.write(f'**Source:**\n{source}')
        st.write(result)
    else:
        st.write('No result to display.')
elif not query:
    st.info('Please enter a question.')
elif not st.session_state.chain:
    st.info('Please add a URL to initialize the chain.')
