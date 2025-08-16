#!/usr/bin/env python3
"""
Test script for local model functionality
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.local_evaluator import initialize_local_model, classify_with_local_model

def test_local_model():
    """Test the local model with a simple example"""
    print("🧪 Testing Local Model Functionality")
    print("=" * 50)
    
    try:
        # Initialize the model
        initialize_local_model()
        
        # Test with a simple conversation
        conversation_history = """
        Student: I wrote "The cat sleep on the mat" but my teacher said it's wrong.
        """
        
        tutor_response = "You need to add 's' to 'sleep' because the cat is singular. It should be 'The cat sleeps on the mat'."
        
        print("\n📝 Testing classification...")
        print(f"Conversation: {conversation_history.strip()}")
        print(f"Tutor Response: {tutor_response}")
        
        # Classify the response
        mi_label, pg_label = classify_with_local_model(conversation_history, tutor_response)
        
        print(f"\n✅ Classification Results:")
        print(f"Mistake Identification: {mi_label}")
        print(f"Providing Guidance: {pg_label}")
        
        print("\n🎉 Local model test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_local_model()
