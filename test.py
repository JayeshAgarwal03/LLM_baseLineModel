#!/usr/bin/env python3
"""
Simple Local LLM Runner
Uses TinyLlama-1.1B-Chat (2-4GB RAM usage)
"""

import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

# Only import BitsAndBytesConfig if CUDA is available
try:
    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        QUANTIZATION_AVAILABLE = True
    else:
        QUANTIZATION_AVAILABLE = False
except ImportError:
    QUANTIZATION_AVAILABLE = False

class SimpleLLM:
    def __init__(self):
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
    
    def load_model(self):
        """Load the TinyLlama model with optimizations"""
        print(f"Loading {self.model_name}...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure model loading based on available hardware
            if self.device == "cuda" and QUANTIZATION_AVAILABLE:
                # Use quantization for GPU
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
            else:
                # CPU-only loading without quantization
                print("Loading for CPU (no quantization)...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,  # Use float32 for CPU
                    trust_remote_code=True
                )
                # Move to CPU explicitly
                self.model = self.model.to('cpu')
            
            print("✓ Model loaded successfully!")
            print(f"Memory usage: ~{self.get_memory_usage():.1f}GB")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def get_memory_usage(self):
        """Get approximate memory usage"""
        if torch.cuda.is_available() and self.device == "cuda":
            return torch.cuda.memory_allocated() / 1024**3
        else:
            return 3.0  # Approximate CPU usage for TinyLlama
    
    def generate(self, prompt, max_tokens=256, temperature=0.7):
        """Generate text from prompt"""
        if self.model is None:
            self.load_model()
        
        # Format prompt for chat model
        formatted_prompt = f"<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
        
        # Tokenize and move to correct device
        inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                attention_mask=torch.ones_like(inputs)
            )
            end_time = time.time()
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs.shape[1]:], 
            skip_special_tokens=True
        )
        
        generation_time = end_time - start_time
        tokens_generated = len(outputs[0]) - len(inputs[0])
        
        return {
            'response': response.strip(),
            'time': generation_time,
            'tokens': tokens_generated,
            'tokens_per_second': tokens_generated / generation_time if generation_time > 0 else 0
        }
    
    def chat(self):
        """Interactive chat mode"""
        print("\n=== TinyLlama Chat ===")
        print("Type 'quit' to exit, 'clear' to clear screen")
        print("-" * 40)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'clear':
                    import os
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                elif not user_input:
                    continue
                
                # Generate response with shorter output for CPU
                result = self.generate(user_input, max_tokens=50)  # Shorter for CPU
                
                print(f"\nBot: {result['response']}")
                print(f"⏱️  {result['time']:.2f}s | {result['tokens_per_second']:.1f} tokens/s")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

def test_model():
    """Test the model with sample prompts"""
    llm = SimpleLLM()
    
    test_prompts = [
        "Hello, how are you today?",
        "What is machine learning?",
        "Write a short poem about coding.",
        "Explain quantum computing in simple terms.",
        "What are the benefits of renewable energy?"
    ]
    
    print("\n=== Model Testing ===")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: {prompt}")
        print("-" * 50)
        
        result = llm.generate(prompt, max_tokens=150)
        
        print(f"Response: {result['response']}")
        print(f"Stats: {result['time']:.2f}s | {result['tokens']} tokens | {result['tokens_per_second']:.1f} tok/s")

def benchmark():
    """Simple benchmark"""
    llm = SimpleLLM()
    
    prompt = "Explain artificial intelligence in one paragraph."
    num_runs = 3
    
    print(f"\n=== Benchmark (average of {num_runs} runs) ===")
    print(f"Prompt: {prompt}")
    print("-" * 50)
    
    times = []
    tokens_per_sec = []
    
    for i in range(num_runs):
        print(f"Run {i+1}...")
        result = llm.generate(prompt, max_tokens=100)
        times.append(result['time'])
        tokens_per_sec.append(result['tokens_per_second'])
    
    avg_time = sum(times) / len(times)
    avg_tps = sum(tokens_per_sec) / len(tokens_per_sec)
    
    print(f"\nResults:")
    print(f"Average time: {avg_time:.2f}s")
    print(f"Average speed: {avg_tps:.1f} tokens/s")
    print(f"Memory usage: ~{llm.get_memory_usage():.1f}GB")

def main():
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            test_model()
        elif command == "benchmark":
            benchmark()
        elif command == "chat":
            llm = SimpleLLM()
            llm.chat()
        else:
            print("Unknown command. Use: test, benchmark, or chat")
    else:
        # Default: interactive mode
        print("Simple Local LLM Runner")
        print("Available commands:")
        print("  python simple_llm.py test      - Run tests")
        print("  python simple_llm.py benchmark - Run benchmark")
        print("  python simple_llm.py chat      - Interactive chat")
        
        choice = input("\nWhat would you like to do? (test/benchmark/chat): ").strip().lower()
        
        if choice == "test":
            test_model()
        elif choice == "benchmark":
            benchmark()
        elif choice == "chat":
            llm = SimpleLLM()
            llm.chat()
        else:
            print("Starting chat mode...")
            llm = SimpleLLM()
            llm.chat()

if __name__ == "__main__":
    main()