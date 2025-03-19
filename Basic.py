# taken from langchain version 3
# both chatpromplate
# Basic final code without UI
import os
import pickle
import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
load_dotenv()

# Load environment variables
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
# Initialize components
llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.1-8b-instant", temperature=0.2)


# Initialize improved embeddings
hf_embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=huggingface_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2")#BAAI/bge-base-en-v1.5

# File paths
pdf_path_new = r"C:\Users\lenovo\Desktop\Project\Agentic_ai\File\modified_CSCRF_sebi_circular_august_20_2024.pdf"
faiss_index_path = "Faiss_Index_V4"

# Enhanced prompt for contextualizing questions based on chat history
contextualize_q_prompt  = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt_template  = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")])

# Enhanced system prompt for response generation
system_prompt = """
You are an expert Information Retrieval and Synthesis System. 
Your role is to extract and synthesize relevant information from the provided context with precision, ensuring **completeness over conciseness** while avoiding redundancy.

Context: {context}

Instructions:
1. **Analyze the question thoroughly** to identify all relevant information from the provided context.
2. **Extract and synthesize information comprehensively**—include all necessary details while ensuring each unique piece of information is mentioned only once.
3. **Avoid repetition**—if a requirement or standard appears multiple times in the context, mention it only once in a structured manner.
4. **Group similar points together** to provide a structured and well-organized answer, reducing redundancy while enhancing clarity.
5. **If multiple sources mention the same detail, consolidate them** instead of repeating them verbatim.
6. **Prioritize completeness over conciseness**—your answer must capture all relevant aspects, even if it results in a longer response.
7. **If any crucial information appears to be missing from the context**, explicitly state: 
   - *"The provided context does not contain information regarding [missing detail]. Additional information may be needed to provide a complete answer."*
8. **Base your response strictly on the provided context**—do not add any external knowledge, assumptions, or hallucinated information.
9. **If the question has multiple sub-questions, address each one explicitly** to ensure no aspect is left unanswered.
10. **Structure your response logically**—use bullet points, numbered lists, or sections to enhance readability while maintaining detail.

Your goal is to provide an exhaustive, well-structured, and **non-repetitive** synthesis of the provided information, ensuring every relevant aspect is covered.
"""
# Response generation chain
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# Function to load or create FAISS index
def load_or_create_faiss_index(pdf_path_new, faiss_index_path):
    if os.path.exists(faiss_index_path):
        print("Loading existing FAISS index...")
        with open(faiss_index_path, "rb") as f:
            db = pickle.load(f)
        return db
    else:
        print("Creating new FAISS index...")

        # Load and process PDF
        loader = PyPDFLoader(file_path=pdf_path_new)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
        split_documents = text_splitter.split_documents(documents)
        # Create FAISS index
        db = FAISS.from_documents(split_documents, hf_embeddings)
        
        # Save FAISS index
        with open(faiss_index_path, "wb") as f:
            pickle.dump(db, f)
        
        print("FAISS index created and saved.")
        return db
# Load or create FAISS index
db = load_or_create_faiss_index(pdf_path_new, faiss_index_path)

# Configure advanced retriever with hybrid search and reranking
retriever=db.as_retriever(search_kwargs={'k':15,'fetch_k': 30},lambda_mult=0.7,search_type="mmr")

# Create history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt_template 

)

# Create final chain with document handling
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
# Store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
question = st.text_input("Ask a financial question:")

if st.button("Submit") and question.strip():
    with st.spinner("Generating response..."):
        response = rag_chain.invoke({"input": question, "chat_history": st.session_state.chat_history})
        answer = response["answer"]

        # Update chat history
        st.session_state.chat_history.append(("human", question))
        st.session_state.chat_history.append(("ai", answer))

        # Display answer
        st.subheader("Answer:")
        st.write(answer)

# Demo function to show interaction
# def process_query(question, chat_history):
#     response = rag_chain.invoke({
#         "input": question, 
#         "chat_history": chat_history
#     })
    
#     return response["answer"], chat_history


# chat_history = []

# while True:

#     question = str(input("Query: "))
#     if question.lower() =='exit':
#         break

#     answer, chat_history = process_query(question, chat_history)

#     chat_history.append(("human", question))
#     chat_history.append(("ai", answer))
#     print("Answe: ",answer)


