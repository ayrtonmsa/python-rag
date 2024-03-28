import streamlit as st

from langchain_community.document_loaders import WebBaseLoader

from langchain_community.vectorstores import Chroma

from langchain_community.llms import Ollama

from langchain_core.runnables import RunnablePassthrough

from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate

from langchain.text_splitter import CharacterTextSplitter

from langchain_community.embeddings import OllamaEmbeddings

# URL processing
def process_input(urls, question):
    model_local = Ollama(model="mistral")
    retriever = ""

    if (urls):
        # Convert string of URLs to list

        urls_list = urls.split("\n")
        docs = [WebBaseLoader(url).load() for url in urls_list]
        docs_list = [item for sublist in docs for item in sublist]

        #split the text into chunks
        
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
        doc_splits = text_splitter.split_documents(docs_list)
        
        #convert text chunks into embeddings and store in vector database

        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=OllamaEmbeddings(model='nomic-embed-text'),
        )
        retriever = vectorstore.as_retriever()
    
    #perform the RAG 
    if (retriever == ""):
        after_rag_template = """Answer the question:
        Question: {question}
        """
        context_question = {"question": RunnablePassthrough()}
    else:
        after_rag_template = """Answer the question based only on the following context:
        {context}
        Question: {question}
        """
        context_question = {"context": retriever, "question": RunnablePassthrough()}

    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        context_question
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    return after_rag_chain.invoke(question)

st.title("Document Query with Ollama")
st.write("Enter URLs (one per line) and a question to query the documents.")

# Input fields
urls = st.text_area("Enter URLs separated by new lines", height=150)
question = st.text_input("Question")

# Creating two columns of equal width
col1, col2 = st.columns(2)

# Button to process input
def buttonDocuments():
    if st.button('Query Documents'):
        with st.spinner('Processing...'):
            answerDocuments = process_input(urls, question)
            return answerDocuments
    return None
def buttonModel():
    if st.button('Query Model'):
        with st.spinner('Processing...'):
            answerFromModel = process_input("", question)
            return answerFromModel
    return None


with col1:
    answerDocuments = buttonDocuments()
with col2:
    answerModel = buttonModel()

if (answerDocuments):
    st.text_area("Answer Documents", value=answerDocuments, height=300, disabled=True)

if (answerModel):
    st.text_area("Answer Model", value=answerModel, height=300, disabled=True)