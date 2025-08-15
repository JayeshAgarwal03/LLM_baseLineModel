import argparse
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import evaluators
from src.models.gemini_evaluator import run_gemini_evaluation
from src.models.groq_evaluator import run_groq_evaluation

# -------------------------------
# MAIN CONTROLLER
# -------------------------------
# This file serves as the main controller that handles command line arguments
# and calls the appropriate model evaluator based on user selection.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM Baseline Model Evaluation')
    parser.add_argument('--model', '-m', 
                       choices=['gemini', 'groq'], 
                       required=True,
                       help='Choose which model to evaluate: gemini or groq')
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
    else:
        print(f"Invalid model choice: {args.model}")
        print("Please choose 'gemini' or 'groq'")
