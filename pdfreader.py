#first install these library 
# pip install langchain openai PyPDF2 faiss-cpu tiktoken langchain_community
  

#And then you import these
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

import os
# from google.colab import userdata
# google_api_key=userdata.get('.......Paste API KEY........')
os.environ["OPENAI_API_KEY"] = '.......Paste API KEY........'

pdfreader=PdfReader('Budget_in_Brief.pdf')

from typing_extensions import Concatenate
raw_text=''
for i,page in enumerate(pdfreader.pages):
  content=page.extract_text()
  if content:
    raw_text+=content

#

raw_text

text_spliter=CharacterTextSplitter(
    separator='\n',
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts=text_spliter.split_text(raw_text)

len(texts)

embeding=OpenAIEmbeddings()

documents_search=FAISS.from_texts(texts,embeding)

documents_search

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

chain=load_qa_chain(OpenAI(),chain_type='stuff')

query='Public Financial Management & Accountability (Prov inces-P4R)'
docs=documents_search.similarity_search(query)
chain.run(input_documents=docs,question=query)

