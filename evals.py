import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Tuple
import time

from classify_email import classify_email
from providers import OpenAIProvider, OllamaProvider, DistilBertProvider

def load_test_data(db_path) -> List[Tuple[str, str]]:
    """
    Load test examples from the SQLite database.
    Returns list of (content, expected_label) tuples.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('SELECT body, sender_name, sender_email, subject, label FROM labeled_emails')
    examples = cursor.fetchall()
    
    conn.close()
    return examples

def evaluate_classifier(db_path, provider) -> Dict:
    """
    Run evals using examples from the SQLite database.
    """
    
    examples = load_test_data(db_path)
    
    # Metrics for scoring
    total = len(examples)
    correct = 0
    results = []
    
    print(f"Evaluating {total} examples...")

    # Run evaluation
    for body, sender_name, sender_email, subject, expected in examples:
        content = f"From: {sender_name} <{sender_email}>\nSubject: {subject}\n\n{body[:1000]}"

        predicted = classify_email(provider, content)
        is_correct = predicted == expected
        
        if is_correct:
            print("+", end="")
            correct += 1
        else:
            print("-", end="")
            
        results.append({
            'content': content[:100] + '...' if len(content) > 100 else content,
            'expected': expected,
            'predicted': predicted,
            'correct': is_correct
        })
    
    # Score it
    accuracy = correct / total if total > 0 else 0
    
    return {
        'total_examples': total,
        'correct_predictions': correct,
        'accuracy': accuracy,
        'detailed_results': results
    }

if __name__ == "__main__":
    db_path = 'datasets/for-evals.sqlite'
    
    # provider = OpenAIProvider(model="gpt-4o")
    # provider = OpenAIProvider(model="gpt-4o-mini")
    # provider = OllamaProvider(model="llama3.2")
    # provider = DistilBertProvider()
    provider = OpenAIProvider(model="ft:gpt-4o-mini-2024-07-18:personal:emailtriage:Ajlgwmgb")

    start_time = time.time()
    
    try:
        results = evaluate_classifier(db_path, provider=provider)
        
        elapsed_time = time.time() - start_time
        
        print(f"\nEvaluation Results:")
        print(f"Total examples: {results['total_examples']}")
        print(f"Correct predictions: {results['correct_predictions']}")
        print(f"Accuracy: {results['accuracy']:.2%}")
        print(f"Total time: {elapsed_time:.2f} seconds")
    
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
