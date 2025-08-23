import argparse
import os
from dotenv import load_dotenv

load_dotenv()

from src.models.gemini_evaluator import run_gemini_evaluation
from src.models.groq_evaluator import run_groq_evaluation
from src.models.TinyLlama_evaluator import run_tinyllama_evaluation
from src.models.hugging_face_evaluator import run_huggingface_evaluation # Import the new function

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM Baseline Model Evaluation')
    parser.add_argument('--model', '-m', 
                       choices=['gemini', 'groq', 'tinyllama', 'huggingface'], 
                       required=True,
                       help='Choose which model to evaluate: gemini, groq, tinyllama, or huggingface')
    parser.add_argument('--dataset', '-d',
                       required=True,
                       help='Path to the dataset JSON file')
    
    args = parser.parse_args()
    
    print("LLM Baseline Model Evaluation")
    print(f"Selected model: {args.model.upper()}")
    print(f"Dataset path: {args.dataset}")
    
    if args.model.lower() == 'gemini':
        run_gemini_evaluation(args.dataset)
    elif args.model.lower() == 'groq':
        run_groq_evaluation(args.dataset)
    elif args.model.lower() == 'tinyllama':
        run_tinyllama_evaluation(args.dataset)
    elif args.model.lower() == 'huggingface':
        run_huggingface_evaluation(args.dataset) # Call the new function
    else:
        print(f"Invalid model choice: {args.model}")
        print("Please choose 'gemini', 'groq', 'tinyllama', or 'huggingface'")