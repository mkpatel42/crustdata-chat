# CrustData Chat Bot

## Introduction

CrustData Chat Bot is an interactive question-answering system that allows users to query information about CrustData. This application uses advanced natural language processing techniques to provide accurate and contextual responses.

## Features

- Interactive chat interface
- Context-aware responses
- Integration with Groq's powerful language model
- Efficient document retrieval using FAISS
- Conversation history for follow-up questions

## Technologies

- Python 3.8+
- Streamlit
- LangChain
- Groq API
- FAISS
- HuggingFace Transformers

## Installation

1. Clone the repository:

2. Install the required packages:


## Usage

1. Set up your Groq API key:
- You'll be prompted to enter your Groq API key in the Streamlit sidebar when you run the app.

2. Run the Streamlit app:


3. Open your web browser and navigate to the URL provided by Streamlit (typically http://localhost:8501).

4. Enter your questions about CrustData in the text input field and receive answers from the chatbot.

5. You can ask follow-up questions, and the chatbot will maintain context from previous interactions.

## Configuration

The application uses the following main components:

- Groq's `mixtral-8x7b-32768` model for language processing
- FAISS for efficient vector storage and retrieval
- HuggingFace's `all-MiniLM-L6-v2` model for text embeddings

These can be adjusted in the code if needed.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Troubleshooting

If you encounter any issues:
1. Ensure your Groq API key is entered correctly.
2. Check your internet connection, as the app requires access to external APIs.
3. Make sure all dependencies are correctly installed.

For any persistent problems, please open an issue on the GitHub repository.
