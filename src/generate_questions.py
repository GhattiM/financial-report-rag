import json
import os
from query import setup_rag_chain

def generate_new_questions():
    print("🤖 Setting up RAG chain for question generation...")
    rag_chain = setup_rag_chain()
    
    # Topics to explore for new questions
    topics = [
        "What are the total assets of Apple vs Amazon?",
        "How much did NVIDIA spend on inventory in their latest fiscal year?",
        "What is the total debt (long-term and short-term) for Amazon?",
        "Compare the 'Cost of Sales' for Apple and NVIDIA.",
        "What are the main product categories for Apple and their respective revenues?",
        "Describe the 'Segment Operating Income' for Amazon's AWS business.",
        "What was NVIDIA's 'Gross Margin' percentage in the latest year?",
        "Identify any significant 'Commitments and Contingencies' mentioned by Apple.",
        "What is the 'Weighted Average Shares Outstanding' for Amazon (basic and diluted)?",
        "Compare the 'Property, Plant, and Equipment, Net' for all three companies."
    ]
    
    new_entries = []
    
    for q in topics:
        print(f"\n🔍 Querying: {q}")
        try:
            response = rag_chain.invoke({"input": q})
            answer = response['answer']
            context_docs = response['context']
            
            # Extract basic info for the golden dataset format
            # We'll use the LLM's answer as the ground truth for now, 
            # assuming the RAG chain is accurate enough for this task.
            
            # Simplified source extraction
            sources = ", ".join(list(set([
                f"{doc.metadata.get('source', 'Unknown')} p.{doc.metadata.get('page', '?')}" 
                for doc in context_docs[:3] # top 3 sources
            ])))
            
            company = "Multi"
            if "Apple" in q and "Amazon" not in q and "NVIDIA" not in q:
                company = "Apple"
            elif "Amazon" in q and "Apple" not in q and "NVIDIA" not in q:
                company = "Amazon"
            elif "NVIDIA" in q and "Apple" not in q and "Amazon" not in q:
                company = "NVIDIA"
                
            new_entries.append({
                "question": q,
                "answer": answer,
                "source": sources,
                "company": company
            })
            print("✅ Captured.")
        except Exception as e:
            print(f"❌ Error: {e}")

    # Load existing
    dataset_path = 'data/golden_dataset.json'
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = []
        
    # Append 10 new ones (or however many we got)
    existing_data.extend(new_entries[:10])
    
    with open(dataset_path, 'w') as f:
        json.dump(existing_data, f, indent=4)
        
    print(f"\n🚀 Successfully added {len(new_entries)} questions to {dataset_path}")

if __name__ == "__main__":
    generate_new_questions()
