import gc
import os
import shutil
import time
import torch
import requests
from typing import Sequence, Any, Optional
from operator import itemgetter
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.chains import create_retrieval_chain
from langchain_community.document_transformers import LongContextReorder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.callbacks import Callbacks
from langchain_core.documents.compressor import BaseDocumentCompressor
import ingest
from langchain_openai import ChatOpenAI

# 1. Setup API Key 
load_dotenv()

# Automatic device detection
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔌 Using device: {device}")

# Enable GPU for embeddings only if available
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'device': device}
)

# 1. Initialize the Reorderer
reorderer = LongContextReorder()

# 2. Define a helper function to reorder docs
def reorder_documents(docs):
    return reorderer.transform_documents(docs)

def load_system_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Custom OpenRouter Reranker Implementation
class OpenRouterRerank(BaseDocumentCompressor):
    model: str = "cohere/rerank-4-pro"
    top_n: int = 12
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY")

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []
        
        url = "https://openrouter.ai/api/v1/rerank"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare documents for the API
        doc_list = [doc.page_content for doc in documents]
        
        payload = {
            "model": self.model,
            "query": query,
            "documents": doc_list,
            "top_n": self.top_n
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code != 200:
            print(f"❌ Rerank Error: {response.status_code} - {response.text}")
            return documents[:self.top_n] # Fallback to top_n original
            
        data = response.json()
        
        # OpenRouter/Cohere format: results: [{index: 0, relevance_score: 0.9}, ...]
        results = data.get("results", [])
        
        final_docs = []
        for res in results:
            idx = res.get("index")
            if idx is not None and idx < len(documents):
                doc = documents[idx]
                doc.metadata["relevance_score"] = res.get("relevance_score")
                final_docs.append(doc)
                
        return final_docs

def setup_rag_chain():
    print("🧠 Loading AI Brain...")

    # 0. Setup LLM for final answer using OpenRouter
    llm = ChatOpenAI(
        model="google/gemini-2.0-flash-001",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0
    )
    
    # Identify all PDF sources - handle paths relative to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "10k")
    
    # Get PDFs from data/10k
    pdf_files_data = []
    if os.path.exists(data_dir):
        pdf_files_data = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    
    all_pdf_paths = pdf_files_data
    
    if not all_pdf_paths:
        print("⚠️ No PDF files found! Please check your data directories.")
        return None
        
    folder_sources = {os.path.basename(f).split("_")[0].upper() for f in all_pdf_paths}
    
    # Check if chromadb exists and has all the data
    should_reprocess = True
    chroma_path = os.path.join(base_dir, "chromadb")
    db_file = os.path.join(chroma_path, "chroma.sqlite3")
    
    if os.path.exists(db_file):
        print("📦 Checking existing database via sqlite...")
        import sqlite3
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT string_value FROM embedding_metadata WHERE key='source';")
            db_sources = {row[0] for row in cursor.fetchall() if row[0]}
            conn.close()
            
            if folder_sources.issubset(db_sources) and len(db_sources) > 0:
                print(f"✅ Database already contains data for: {', '.join(db_sources)}")
                should_reprocess = False
                vectorstore = Chroma(
                    persist_directory=chroma_path,
                    embedding_function=embeddings
                )
            else:
                print(f"⚠️ Database incomplete or missing. Expected {folder_sources}, found {db_sources}. Re-ingesting...")
        except Exception as e:
            print(f"⚠️ Error checking database: {e}. Re-ingesting...")
    
    if should_reprocess:
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)
            
        print(f"🚀 Starting fresh ingestion of {len(all_pdf_paths)} files...")
        vectorstore = ingest.process_pdfs_from_list(all_pdf_paths)
    
    # 2. Setup Hybrid Search
    all_content = vectorstore.get()
    
    # Efficient keyword search
    keyword_retriever = BM25Retriever.from_texts(
        all_content['documents'], 
        metadatas=all_content['metadatas']
    )
    keyword_retriever.k = 50  # Increased for better recall
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 50})

    # Hybrid blend
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, keyword_retriever], 
        weights=[0.5, 0.5] # Balanced for financial data
    )

    # 3. Using Custom OpenRouter Reranker
    compressor = OpenRouterRerank(
        model="cohere/rerank-v3.5", # Use the latest reranker model name
        top_n=12
    )
    
    # Wrap the hybrid retriever with the reranker
    from langchain_classic.retrievers import ContextualCompressionRetriever
    reranked_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=hybrid_retriever
    )

    def get_filtered_retriever(company_name, query):
        """Creates a company-specific hybrid retriever."""
        # 1. Vector with hard filter
        v_ret = vectorstore.as_retriever(search_kwargs={
            "k": 30, 
            "filter": {"source": company_name}
        })
        
        # 2. BM25 with manual filter (Subset the documents)
        indices = [i for i, m in enumerate(all_content['metadatas']) if m.get('source') == company_name]
        company_docs = [all_content['documents'][i] for i in indices]
        company_metas = [all_content['metadatas'][i] for i in indices]
        
        if not company_docs:
            return v_ret
            
        b_ret = BM25Retriever.from_texts(company_docs, metadatas=company_metas)
        b_ret.k = 30
        
        return EnsembleRetriever(retrievers=[v_ret, b_ret], weights=[0.5, 0.5])

    # 4. Multi-Company Strategy
    from langchain_core.runnables import RunnableLambda
    
    # 🔍 DYNAMIC COMPANY DETECTION
    db_file = os.path.join(base_dir, "chromadb", "chroma.sqlite3")
    cached_db_sources = []
    if os.path.exists(db_file):
        import sqlite3
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            # Try to get unique sources from the DB
            cursor.execute("SELECT DISTINCT string_value FROM embedding_metadata WHERE key='source';")
            cached_db_sources = [row[0].upper() for row in cursor.fetchall() if row[0]]
            conn.close()
        except Exception:
            pass
    
    # Fallback to file system if DB is empty/unavailable
    if not cached_db_sources:
        pdf_dir = os.path.join(base_dir, "data", "10k")
        if os.path.exists(pdf_dir):
            cached_db_sources = [f.split("_")[0].upper() for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

    def expand_query(query):
        """Expands the query with common financial synonyms to improve BM25 recall."""
        synonyms = {
            "R&D": "Research and Development, technology and infrastructure, technology and content",
            "NET INCOME": "Net income, net earnings, earnings, net profit",
            "REVENUE": "Net sales, revenue, total sales, product sales, service sales",
            "CAPEX": "Purchases of property and equipment, capital expenditures",
            "SHARES OUTSTANDING": "Common shares outstanding, shares issued and outstanding, weighted average shares",
            "EMPLOYEES": "Full-time employees, number of employees, headcount",
            "INCOME TAX": "Provision for income taxes, income tax expense",
            "CASH": "Cash and cash equivalents, marketable securities, liquidity",
            "INCORPORATION": "State of incorporation, jurisdiction of incorporation, registrant",
        }
        
        expanded = query
        q_upper = query.upper()
        for key, value in synonyms.items():
            if key in q_upper:
                expanded += f" ({value})"
        return expanded

    def multi_company_search(query_input):
        q_upper = query_input.upper()
        # Expand the query for better matching
        expanded_q = expand_query(query_input)
        
        found_companies = [c for c in cached_db_sources if c in q_upper]
        
        # Generic comparative keywords
        compare_keywords = ["ALL", "EACH", "COMPARE", "DIFFERENCE", "MOST", "LEAST", "HIGHEST", "LOWEST", "COMBINED", "TOTAL", "LIST"]
        is_generic_compare = any(k in q_upper for k in compare_keywords)
        
        if is_generic_compare and not found_companies:
            found_companies = cached_db_sources
            print(f"🏢 Generic comparison. Searching: {', '.join(found_companies)}")

        if not found_companies:
            return reranked_retriever.invoke(expanded_q)
            
        print(f"🏢 Targeted Hybrid search for: {', '.join(found_companies)}")
        all_docs = []
        
        # 1. Check if we need cover page info
        cover_page_keywords = ["INCORPORATION", "HEADQUARTERS", "ADDRESS", "EXCHANGE", "TICKER", "SYMBOL", "STATE OF", "FISCAL YEAR END", "REGISTRANT", "IRS", "COVER", "EMPLOYEE", "SHARES OUTSTANDING"]
        needs_cover = any(k in q_upper for k in cover_page_keywords)

        # INCREASED PRECISION: Use hard metadata filters per company
        for company in found_companies:
            # 1. Retrieve cover pages (Page 1-5) if needed
            if needs_cover:
                print(f"📄 Corporate fact keywords detected. Adding pages 1-5 for {company}...")
                for page_num in range(1, 6):
                    cover_docs = vectorstore.similarity_search(
                        f"{company} corporate information legal entity shares employees", 
                        k=1, 
                        filter={"$and": [{"source": company}, {"page": page_num}]}
                    )
                    all_docs.extend(cover_docs)

            # 2. Targeted Hybrid search for the actual company
            f_retriever = get_filtered_retriever(company, expanded_q)
            company_docs = f_retriever.invoke(expanded_q)
            
            # Rerank these specific docs for the current company once
            if company_docs:
                reranked_company_docs = compressor.compress_documents(company_docs, expanded_q)
                all_docs.extend(reranked_company_docs[:12]) # Keep top 12 per company
            
        # Deduplicate
        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen:
                unique_docs.append(doc)
                seen.add(doc.page_content)
                
        return unique_docs

    prompt_path = os.path.join(base_dir, "src", "system_prompt.md")
    if not os.path.exists(prompt_path):
        prompt_path = os.path.join(base_dir, "system_prompt.md")
        
    system_prompt = load_system_prompt(prompt_path)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question: {input}"),
    ])
    
    # Create the modern retrieval chain
    rag_chain = (
        {
            "context": itemgetter("input") | RunnableLambda(multi_company_search) | reorder_documents, 
            "input": itemgetter("input")
        }
        | RunnablePassthrough.assign(
            answer=(
                prompt 
                | llm 
                | StrOutputParser()
            )
        )
    )
    
    return rag_chain

def ask_finance(rag_chain, question):
    print(f"\n🔍 Searching 10-Ks for: {question}")
    
    # 1. Get the result from the chain
    result = rag_chain.invoke({"input": question})
    
    # 2. Print the answer
    print("\n" + "🤖 AI ANSWER:" + "\n" + "-"*15)
    print(result['answer'].strip())
    
    # 3. Print the sources
    print("\n📚 SOURCES USED:")
    sources = sorted(list(set([
        f"- {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', '?')})" 
        for doc in result['context']
    ])))
    
    for s in sources:
        print(s)
        
    return result

if __name__ == "__main__":
    chain = setup_rag_chain()
    
    print("\n" + "="*50)
    print("🚀 FINANCIAL RAG TERMINAL READY")
    print("="*50)

    while True:
        print("\n" + "—"*50)
        user_query = input("💬 Question: ").strip()

        if user_query.lower() in ['exit', 'quit', 'q']:
            print("👋 Closing session.")
            break
        
        if user_query:
            ask_finance(chain, user_query)
        else:
            print("⚠️ Please enter a question.")
