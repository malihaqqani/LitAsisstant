import json
import numpy as np
import pandas as pd
from datasets import load_dataset
from ragagent.agent.enhanced_graph import run_enhanced_supervisor_workflow

# Evaluation on a subset of LitQA2
litqa = load_dataset("lit-qa", "litqa2", split="test[:50]")  # first 50 examples

results = []

for example in litqa:
    question = example["question"]
    gold_para_ids = example["paragraphs"]  # list of gold paragraph IDs
    # Run the agentic RAG pipeline
    output = run_enhanced_supervisor_workflow(question)
    # Extract processed paper fragments and logs
    processed = output.get("processed_papers", [])
    # For retrieval metric, check top-k overlap with gold paragraphs
    retrieved_ids = [getattr(p, "paper_id", None) for p in processed]
    # Compute Precision@k, Recall@k
    k = len(processed)
    overlap = len(set(retrieved_ids) & set(gold_para_ids))
    precision = overlap / k if k else 0
    recall = overlap / len(gold_para_ids) if gold_para_ids else 0
    
    # Dummy answer accuracy (placeholder)
    # In practice, compare output["final_answer"] to example["answer"]
    answer_correct = np.random.choice([0,1])  # placeholder random
    
    results.append({
        "question": question,
        "precision@k": precision,
        "recall@k": recall,
        "answer_accuracy": answer_correct
    })

df_results = pd.DataFrame(results)
# Compute aggregate metrics
aggregate = {
    "prec@avg": df_results["precision@k"].mean(),
    "rec@avg": df_results["recall@k"].mean(),
    "answer_acc": df_results["answer_accuracy"].mean()
}

import ace_tools as toooils; tools.display_dataframe_to_user(name="LitQA2 Evaluation Results", dataframe=df_results)
print("Aggregate Metrics:", aggregate)
