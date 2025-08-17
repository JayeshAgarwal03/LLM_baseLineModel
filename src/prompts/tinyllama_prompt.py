# -------------------------------
# TINYLLAMA-SPECIFIC PROMPT TEMPLATE
# -------------------------------

TINYLLAMA_CLASSIFICATION_PROMPT = """You are a helpful assistant that evaluates tutor responses. You must respond with exactly two lines in this format:

Mistake Identification: [Yes/No/To some extent]
Providing Guidance: [Yes/No/To some extent]

Rules:
- Choose "Yes" if the tutor clearly identified the mistake or provided helpful guidance
- Choose "No" if the tutor did not identify the mistake or failed to provide guidance  
- Choose "To some extent" if the tutor partially identified the mistake or gave incomplete guidance
- Do not add any other text or explanations
- Use exactly the format above

Conversation History:
{conversation_history}

Tutor Response:
{tutor_response}

Your evaluation:"""

# Alternative simpler prompt for TinyLlama
TINYLLAMA_SIMPLE_PROMPT = """Answer in exactly this format:
Mistake Identification: [Yes/No/To some extent]
Providing Guidance: [Yes/No/To some extent]

Conversation: {conversation_history}
Tutor: {tutor_response}"""
