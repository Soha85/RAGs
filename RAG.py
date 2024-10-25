import pandas as pd
import re
class RAG:
    articles = pd.DataFrame([])
    corpus_chunks = []
    def __init__(self):
        self.articles['all_content'] = [self.preprocess_text(row['title'] + " " + row['content']) for x, row in self.articles.iterrows()]

    def prepare_data(self,chunk_size,overlap):
        for context in self.articles["all_content"]:
            # Combine question and context (as one block of text)
            # Split the document into chunks
            chunks = self.chunk_text(context,chunk_size,overlap)
            self.corpus_chunks.extend(chunks)  # Add chunks to the corpus


    def preprocess_text(self,text):
        text = text.lower()  # Convert to lowercase
        text = text.replace('\\n', '')
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        # text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and digits
        return text.strip()


    # Chunking: Split long documents into smaller chunks
    def chunk_text(self,text, chunk_size, overlap):
        words = text.split(' ')
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
        return chunks