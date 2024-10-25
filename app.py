import streamlit as st
import logging
import requests
from bs4 import BeautifulSoup
import pandas as pd
from transformers import pipeline
import rouge
from RAG import RAG
from NaiveRAG import NaiveRAG
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
st.session_state.articles_df = RAG.articles

def evaluate_rouge(answer,reference):
    if answer:
        evaluator = rouge.Rouge()
        return evaluator.get_scores(answer, reference)
    else:
        return "no score"

# Function to scrape article URLs from a website
def scrape_articles(site_url):
    try:
        response = requests.get(site_url, timeout=10)  # Add a timeout for safety
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx and 5xx)
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching the article: {e}")
        return None, f"Error fetching the article: {e}"

    soup = BeautifulSoup(response.content, 'html.parser')
    # Extract text within the article (based on inspection)
    div_elements = soup.find_all('div', {'data-component': 'text-block'})
    all_paragraphs = []
    title = soup.title.get_text()
    for div in div_elements:
        paragraphs = div.find_all('p')
        for para in paragraphs:
            all_paragraphs.append(para.get_text())

    full_text = ' '.join(all_paragraphs)
    return title, full_text.strip()

def rag_generate(query,context,temperature):
    try:
        articles_llm = pipeline(task='text-generation', model=selected_model)
        articles_llm.model.config.pad_token_id = articles_llm.model.config.eos_token_id
        generated = articles_llm(f"Query: {query}\nContext: {context}\nAnswer:",max_new_tokens=150,temperature=temperature,num_return_sequences=1)
        return generated[0]['generated_text'].split('Answer:')[1]

    except Exception as e:
        st.write(f"Error generating text: {e}")
        return None

# Streamlit UI
st.title("Different RAGs Approaches")
# Session state for storing scraped data
if "articles_df" not in st.session_state:
    st.session_state.articles_df = pd.DataFrame(columns=["title", "content"])

# Initialize previous_website in session state
if "previous_website" not in st.session_state:
    st.session_state.previous_website = None
# Dropdown to select website
selected_website = st.selectbox("Select a website to scrape", ['https://www.bbc.com/travel', 'https://www.bbc.com/culture'])

# Button to get articles
if st.button('Get Articles'):
    article_links = []
    titles = []
    articles = []

    try:
        response = requests.get(selected_website, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        for link in soup.find_all('a', href=True):
            href = link['href']
            if '/article/' in href:
                full_url = 'https://www.bbc.com' + href
                article_links.append(full_url)

        article_links = list(set(article_links))  # Remove duplicates

        for article in article_links:
            title, content = scrape_articles(article)
            if title and content:
                titles.append(title)
                articles.append(content)

        if articles:
            # Display articles in a table
            RAG.articles = pd.DataFrame({'title': titles, 'content': articles})


        st.session_state.articles_df = RAG.articles
        st.success("Articles successfully scraped!")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch articles: {e}")

# Display articles in a table (if any)
if not st.session_state.articles_df.empty:
    st.write(st.session_state.articles_df)
else:
    st.info("No articles scraped yet.")

# Input for user question
question = st.text_input("Ask a question:")
#chunk_size = st.number_input("Chunk Size", min_value=10, max_value=500, value=50, step=50)
#overlap = st.number_input("Overlap Size", min_value=10, max_value=500, value=50, step=10)
#num_answers = st.number_input("Number of Retrievals", min_value=1, max_value=5, value=3, step=1)
temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.1)
models_list = ['gpt2','EleutherAI/gpt-neo-2.7B','EleutherAI/gpt-j-6B','EleutherAI/gpt-neox-20b','t5-large','bigscience/bloom-3b','facebook/opt-6.7b','google/flan-t5-large','meta-llama/LLaMA-7b-hf']
selected_model=st.selectbox("Select a model:",models_list)



# In each column, you can display different RAG outputs
# Button to send the question for processing
if st.button('Ask Question'):

    if not st.session_state.articles_df.empty:
        rag_instance = RAG()
        # Split the down part into three vertical columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.header("Naive RAG")
            try:
                N_RAG = NaiveRAG()
                best_keyword_score, best_matching_record = N_RAG.find_best_match_keyword_search(question, rag_instance.articles["all_content"])
                st.write(f"Keywords matched:{best_matching_record} and its score:{best_keyword_score}")
                score = N_RAG.calculate_cosine_similarity(question, best_matching_record)
                response = best_matching_record
                similarity_score = N_RAG.calculate_enhanced_similarity(question, response)
            except Exception as e:
                st.write(f"Error in {e}")
            st.write("Generated output from RAG model 1")

        with col2:
            st.header("Advanced RAG")
            # Output 2 from RAG goes here
            st.write("Generated output from RAG model 2")

        with col3:
            st.header("Modular RAG")
            # Output 3 from RAG goes here
            st.write("Generated output from RAG model 3")
    else:
        st.error("No articles available for processing.")









# Clear controls when selected website changes
if selected_website != st.session_state.previous_website:
    st.session_state.previous_website = selected_website
    question = ""
    st.empty()
    st.session_state.articles_df = RAG.articles