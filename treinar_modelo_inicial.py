import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle # Para salvar nosso modelo

print("Carregando dados da planilha...")
df = pd.read_excel('lista-pdfs.xlsx')

# Limpando dados (garantir que não há valores nulos que possam quebrar o treino)
df = df.dropna(subset=['Descricao', 'Categoria'])
df['Descricao'] = df['Descricao'].astype(str)

model_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words=None)), # Adicione stopwords em português se desejar
    ('classifier', MultinomialNB())
])

print("Treinando o modelo inicial...")

model_pipeline.fit(df['Descricao'], df['Categoria'])

print("Modelo treinado com sucesso!")


with open('modelo_categoria.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

print("Modelo salvo como 'modelo_categoria.pkl'")