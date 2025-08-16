import json
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.config.local_config import (
    MODEL_NAME,
    TEMPERATURE,
    MAX_NEW_TOKENS,
    TIMEOUT,
    VALID_LABELS,
    NUM_CONVERSATIONS_TO_PROCESS,
    USE_GPU,
    TORCH_DTYPE
)
from src.prompts.prompt import CLASSIFICATION_PROMPT
from src.utils.metrics import display_performance_metrics

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None

def initialize_local_model():
    """
    Initialize the local model and tokenizer
    """
    global model, tokenizer, device
    
    print(f"🚀 GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available() and USE_GPU:
        print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
        device = "cuda"
        torch_dtype = torch.float16 if TORCH_DTYPE == "float16" else torch.float32
    else:
        device = "cpu"
        torch_dtype = torch.float32
        print("⚠️ Using CPU mode")

    print(f"📦 Loading {MODEL_NAME}...")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch_dtype,
                device_map="auto"
            )
            print("✅ Loaded on GPU")
        else:
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            print("✅ Loaded on CPU")

        if torch.cuda.is_available():
            print(f"📊 Memory: ~{torch.cuda.memory_allocated()/1024**3:.1f}GB")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

def parse_local_output(text):
    """
    Parse output like:
    Mistake Identification: Yes
    Providing Guidance: To some extent
    """
    mi_label = None
    pg_label = None
    lines = text.strip().split('\n')
    for line in lines:
        if line.startswith("Mistake Identification:"):
            mi_label = line.split(":", 1)[1].strip()
        elif line.startswith("Providing Guidance:"):
            pg_label = line.split(":", 1)[1].strip()
    return mi_label, pg_label

def classify_with_local_model(conversation_history, tutor_response):
    """
    Classifies a tutor's response using the local model.
    Returns a tuple: (mistake_identification_label, providing_guidance_label)
    """
    global model, tokenizer, device
    
    if model is None or tokenizer is None:
        raise ValueError("Model not initialized. Call initialize_local_model() first.")

    # Format prompt for TinyLlama
    prompt = CLASSIFICATION_PROMPT.format(
        conversation_history=conversation_history,
        tutor_response=tutor_response,
    )
    
    # Format for TinyLlama chat format
    formatted_prompt = f"<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
    
    try:
        print(f"  > Waiting for local model to classify...")
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=4096)
        
        # Move to device if needed
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        end_time = time.time()
        
        # Decode response
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Parse the output
        mi_label, pg_label = parse_local_output(response)
        
        # Validate labels
        if mi_label not in VALID_LABELS:
            print(f"Warning: Mistake Identification label invalid: '{mi_label}'")
            mi_label = "Error"
        if pg_label not in VALID_LABELS:
            print(f"Warning: Providing Guidance label invalid: '{pg_label}'")
            pg_label = "Error"
        
        # Print generation stats
        tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
        speed = tokens_generated / (end_time - start_time) if (end_time - start_time) > 0 else 0
        print(f"  ⏱️ {end_time - start_time:.2f}s | ⚡ {speed:.1f} tok/s")
        
        return mi_label, pg_label

    except Exception as e:
        print(f"An error occurred while calling local model: {e}")
        return "Error", "Error"

def run_local_evaluation(dataset_path):
    """
    Run evaluation using local model
    """
    print("\n" + "="*60)
    print("RUNNING LOCAL MODEL EVALUATION")
    print("="*60)
    
    # Initialize the model
    initialize_local_model()
    
    # Load the dataset
    try:
        with open(dataset_path, 'r') as f:
            dev_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_path}")
        return

    true_labels_mi, predicted_labels_mi = [], []
    true_labels_pg, predicted_labels_pg = [], []

    for conversation in dev_data[:NUM_CONVERSATIONS_TO_PROCESS]:
        conversation_id = conversation['conversation_id']
        conversation_history = conversation['conversation_history']
        tutor_responses = conversation['tutor_responses']

        print(f"\n--- Conversation ID: {conversation_id} ---")

        for tutor_name, response_data in tutor_responses.items():
            tutor_response_text = response_data['response']

            # Ground truth
            true_mi = response_data['annotation']['Mistake_Identification']
            true_pg = response_data['annotation']['Providing_Guidance']
            true_labels_mi.append(true_mi)
            true_labels_pg.append(true_pg)

            # Single prediction returning both labels
            pred_mi, pred_pg = classify_with_local_model(
                conversation_history,
                tutor_response_text
            )
            predicted_labels_mi.append(pred_mi)
            predicted_labels_pg.append(pred_pg)

            print(f"Tutor: {tutor_name}")
            print(f"Response: {tutor_response_text}")
            print(f"Mistake ID -> True: {true_mi}, Predicted: {pred_mi}")
            print(f"Guidance   -> True: {true_pg}, Predicted: {pred_pg}")
            print("-" * 20)

    # Performance metrics
    display_performance_metrics(true_labels_mi, predicted_labels_mi, true_labels_pg, predicted_labels_pg, "Local Model")
