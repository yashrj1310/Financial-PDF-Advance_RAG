# 💼 Financial Document RAG System 💼

## ✨ Overview
This **Financial Document Retrieval-Augmented Generation (RAG) System** is designed to extract exhaustive answers from financial documents, ensuring **completeness over conciseness** while avoiding redundancy. It utilizes **LangChain v3, FAISS, Groq LLM (LLaMA-3.1-8B), and Hugging Face Embeddings** for accurate retrieval and synthesis of information from PDFs.

## 🛠️ Key Features
- **🔄 Memory-Enhanced Retrieval**: Tracks chat history for context-aware responses.
- **🎯 High-Quality Embeddings**: Uses `sentence-transformers/all-MiniLM-l6-v2` for precise similarity search.
- **🏆 Exhaustive Answer Generation**: Ensures all relevant details are included while avoiding redundancy.
- **🔍 FAISS-Based Vector Search**: Enables efficient similarity-based retrieval from financial PDFs.
- **⚙️ MMR Hybrid Search & Reranking**: Balances diversity and relevance in retrieved documents.
- **🏦 UI with Streamlit**: Provides an intuitive interface for querying financial documents.

## ⚡ Tech Stack
- **Programming Language:** Python
- **Frameworks/Libraries:** LangChain v3, FAISS, Open WebUI, Streamlit
- **LLM:** Groq (LLaMA-3.1-8B-Instant)
- **Embeddings:** Hugging Face (`all-MiniLM-l6-v2`)
- **Data Handling:** PyPDFLoader (for financial PDFs)
- **Environment Management:** dotenv (for API keys)

## 📝 Installation & Setup

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/yashrj1310/Financial-Doc-RAG.git
cd Financial-Doc-RAG
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Set Up API Keys
Create a `.env` file and add:
```env
HUGGINGFACE_API_KEY=your_huggingface_api_key
GROQ_API_KEY=your_groq_api_key
```

### 4️⃣ Run the Application
```sh
streamlit run app.py
```

## 🤖 How It Works
1. Upload financial PDFs.
2. The system processes and stores them in a **FAISS vector store**.
3. Users ask questions via the **Streamlit UI**.
4. The **history-aware retriever** refines queries for best retrieval.
5. The **LLM generates an exhaustive, structured response**.

## 🔧 Future Enhancements
- **💡 Docker Support** for easy deployment.
- **🔗 Open WebUI Integration** for querying directly from WebUI.
- **🌟 Advanced Memory Implementation** for better contextual understanding.

## 📊 Example Query
**User:** "What are the key regulations in the August 2024 SEBI circular?"

**System Response:**
```
1. Regulation A: Detailed explanation...
2. Regulation B: Key insights...
...
Missing Information: *The provided document does not cover XYZ regulation.*
```

## 🛠️ Contributing
Pull requests are welcome! Please ensure your code follows best practices.

## 💎 Credits
Developed by **Yash RJ** (@yashrj1310) 🚀

---
**🔗 GitHub Repo:** [Financial-Doc-RAG](https://github.com/yashrj1310/Financial-Doc-RAG)

