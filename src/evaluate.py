import json
import os
import pandas as pd
import nest_asyncio
import torch
from datasets import Dataset
from query import setup_rag_chain
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# Apply nest_asyncio to avoid "asyncio.run() cannot be called from a running event loop"
nest_asyncio.apply()

# Custom wrapper to fix Gemini's illegal JSON escaping (e.g., \') and parsing errors
class CleanChatOpenAI(ChatOpenAI):
    def _generate(self, *args, **kwargs):
        # We override _generate as it is the core method for both invoke and generate
        res = super()._generate(*args, **kwargs)
        for generation in res.generations:
            if hasattr(generation, 'text') and isinstance(generation.text, str):
                text = generation.text.strip()
                
                # 1. Remove markdown code blocks if present
                if text.startswith("```json"):
                    text = text[7:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

                # 2. Fix illegal single quote escaping
                # Gemini often returns \' which is invalid in JSON strings
                text = text.replace("\\'", "'")
                
                # 3. Fix double backslashes
                text = text.replace("\\\\", "\\")

                # 4. Fix potential trailing commas in JSON before the closing brace/bracket
                import re
                text = re.sub(r",\s*}", "}", text)
                text = re.sub(r",\s*]", "]", text)
                
                # 5. Sometimes Gemini adds a leading/trailing quote inside the JSON string
                # which breaks parsing. We try a best-effort fix if it looks like a common pattern.

                generation.text = text
                
                # If the generation message content exists, update it too
                if hasattr(generation, 'message') and hasattr(generation.message, 'content'):
                    generation.message.content = text
        return res

def run_evaluation():
    # 0. Setup Base Directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'data', 'golden_dataset.json')
    output_path = os.path.join(base_dir, 'data', 'eval_ragas_results.json')

    print(f"📍 Project Base: {base_dir}")

    # 1. Load the AI pipeline
    print("🤖 Setting up the RAG chain for evaluation...")
    rag_chain = setup_rag_chain()
    if not rag_chain:
        print("❌ Error: RAG chain setup failed.")
        return

    # 2. Load the golden dataset
    print(f"📂 Loading Golden Dataset from {dataset_path}...")
    try:
        with open(dataset_path, 'r') as f:
            golden_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: '{dataset_path}' not found.")
        return

    # 3. Run RAG chain against the questions
    print(f"🔍 Running RAG chain for {len(golden_data)} questions...")
    
    questions = []
    ground_truths = []
    answers = []
    contexts = []

    for item in golden_data:
        print(f"\n❓ Question: {item['question']}")
        try:
            response = rag_chain.invoke({"input": item['question']})
            
            questions.append(item['question'])
            ground_truths.append(item['answer'])
            answers.append(response['answer'])
            contexts.append([doc.page_content for doc in response['context']])
            
            print("✅ Answered!")
        except Exception as e:
            print(f"❌ Error testing question: {e}")

    # 4. Prepare data for Ragas
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data_dict)

    # 5. Setup Ragas Evaluator
    print("\n⚖️ Starting Ragas Evaluation (Using OpenRouter)...")
    
    evaluator_llm = CleanChatOpenAI(
        model="google/gemini-2.0-flash-001",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

    # Automatic device detection for embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': device}
    )

    # Increased timeout (1000s) and restricted workers (1) for stability
    run_config = RunConfig(timeout=1000, max_workers=1, max_retries=3)

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            answer_correctness,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config
    )

    # 6. Export and Display Results
    df = result.to_pandas()
    df.to_json(output_path, orient='records', indent=4)
    
    print("\n" + "="*50)
    print("📊 RAGAS EVALUATION SUMMARY")
    print("="*50)
    print(result)
    print("="*50)
    print(f"🚀 Detailed results saved to '{output_path}'")

if __name__ == "__main__":
    run_evaluation()
