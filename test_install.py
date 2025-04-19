from transformers import pipeline

def main():
    print("Testing transformers installation...")
    
    try:
        # Create sentiment pipeline
        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        # Test texts
        texts = [
            "I love this product!",
            "This is terrible.",
            "Not bad at all."
        ]
        
        # Run tests
        for text in texts:
            result = classifier(text)
            print(f"\nInput: {text}")
            print(f"Result: {result}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 