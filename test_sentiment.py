from transformers import pipeline

def test_sentiment():
    try:
        # Initialize sentiment analyzer
        sentiment = pipeline(
            task="sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            framework="pt"
        )
        
        # Test with simple input
        text = "I love you"
        result = sentiment(text)
        
        print(f"Input text: {text}")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    print("Testing sentiment analysis...")
    test_sentiment() 