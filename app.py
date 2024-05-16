from flask import Flask, request
from langchain_community.document_loaders import TextLoader
import textwrap
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

app = Flask(__name__)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

def Wrap_Text_preserve_newlines(text, width=200):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text


def runInitialData():
    loader = TextLoader("./static/Bhubaneswar.txt",encoding="utf-8")
    document = loader.load()
    text_Splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_Splitter.split_documents(document)
    embeddings = HuggingFaceEmbeddings()
    embeddeddoc = FAISS.from_documents(docs, embeddings)
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.7, "max_length": 512})
    chain = load_qa_chain(llm, chain_type="stuff")

    return embeddeddoc, chain

embeddeddoc, chain = runInitialData()

@app.route("/")
def hello():
    return "Hi There This is your BOT !"


@app.route("/getAnswer", methods=['POST', 'GET'])
def getAnswer():
    Query =  request.args.get('Question')
    dc = embeddeddoc.similarity_search(Query)
    return chain.run(input_documents=dc, question=Query)

if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=8000)
