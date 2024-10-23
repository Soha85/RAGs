import pandas as pd
class RAG:
    articles = pd.DataFrame([])
    def __init__(self):
        self.articles['all_content'] = [row['title'] + " " + row['content'] for x, row in self.articles.iterrows()]
