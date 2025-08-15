# LLM Baseline Model Evaluation

Automated evaluation of LLM models for tutor response classification using Gemini and Groq APIs.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.example .env
# Edit .env with your actual API keys

# Run evaluation
python main.py --model gemini --dataset data/dev_testset.json
python main.py --model groq --dataset data/dev_testset.json
```


**Arguments:**
- `--model, -m`: Model to evaluate (`gemini` or `groq`)
- `--dataset, -d`: Path to dataset JSON file

## Project Structure

```
src/
├── config/          # API configurations
├── models/          # Model evaluators
├── prompts/         # Evaluation prompts
└── utils/           # Utility functions
```

## Configuration

1. Copy `env.example` to `.env`
2. Set your API keys in the `.env` file:
   ```
   GOOGLE_API_KEY=your_actual_gemini_key
   GROQ_API_KEY=your_actual_groq_key
   ```
3. The `.env` file is automatically ignored by git for security


