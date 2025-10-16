import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Install Hugging Face Transformers
!pip install transformers

from transformers import pipeline

# Load a pre-trained GPT-2 model for text generation
generator = pipeline("text-generation", model="gpt2")

# Try generating text from a prompt
prompt = "Artificial Intelligence will change the world by"
results = generator(prompt, max_length=50, num_return_sequences=1)

logging.info("Generated text:\n")
logging.info(results[0]["generated_text"])
