import requests

def test_server():
    try:
        # Test basic connection
        response = requests.get('http://localhost:8000')
        print(f"Server connection: {'OK' if response.ok else 'Failed'}")
        
        # Test chat endpoint
        chat_response = requests.post(
            'http://localhost:8000/chat',
            json={"user_input": "show me all products"}
        )
        print(f"Chat endpoint: {'OK' if chat_response.ok else 'Failed'}")
        print(f"Response: {chat_response.json()}")
        
    except Exception as e:
        print(f"Error testing server: {str(e)}")

if __name__ == "__main__":
    test_server() 