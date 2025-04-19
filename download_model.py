from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

def download_model():
    # Create models directory
    os.makedirs("./models/t5-small", exist_ok=True)
    
    print("Downloading T5 model and tokenizer...")
    
    # Download model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    
    # Save to local directory
    print("Saving model and tokenizer locally...")
    tokenizer.save_pretrained("./models/t5-small")
    model.save_pretrained("./models/t5-small")
    
    print("Model and tokenizer saved successfully!")

if __name__ == "__main__":
    download_model() 