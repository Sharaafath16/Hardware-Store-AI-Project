import torch
from transformers import pipeline

def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    
    try:
        # Test simple pipeline
        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        result = classifier("Hello, this is a test!")
        print("Pipeline test result:", result)
        
    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    main() 