from transformers import pipeline

# Test basic pipeline
def test_pipeline():
    try:
        classifier = pipeline('sentiment-analysis')
        result = classifier('we love you')
        print("Pipeline test result:", result)
        return True
    except Exception as e:
        print("Error:", str(e))
        return False

# Test model loading
def test_model_loading():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
        
        print("Model loading successful!")
        return True
    except Exception as e:
        print("Model loading error:", str(e))
        return False

if __name__ == "__main__":
    print("Testing transformers installation...")
    pipeline_ok = test_pipeline()
    model_ok = test_model_loading()
    
    if pipeline_ok and model_ok:
        print("All tests passed!")
    else:
        print("Some tests failed. Please check the errors above.") 