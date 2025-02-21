# GradioColpaliChatbot

GradioColpaliChatbot is an AI-powered chatbot that leverages Gradio for the user interface and integrates with Azure for AI responses and PDF document search. This project allows users to interact with the chatbot, ask questions, and receive responses along with relevant images extracted from PDF documents. For Embedding the relevant Images from the documents a local ColPali model is used. So be sure to have a sufficient machine available to run the code.

## Features

- AI-powered chatbot using Azure AI services and ColPali
- PDF document indexing and search using ColPali
- Image extraction from PDF documents and question-answering based on the image contents
- Preview of the three most relevant pages for the chatbot answer
- Logging of query results with HTML output

## Requirements

- Python 3.8+
- Required Python packages (listed in `requirements.txt`)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/tinomager/GradioColpaliChatbot.git
    cd GradioColpaliChatbot
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up environment variables by copying [sample.env](http://_vscodecontentref_/0) to [.env](http://_vscodecontentref_/1) and updating the values:
    ```sh
    cp sample.env .env
    ```

## Configuration

## Configuration Parameters

The following environment variables need to be set in the `.env` file for the GradioColpaliChatbot to function correctly. Use the `sample.env` file as a starting point for configuration and copy the file as `.env` and fill in your specific configuration:

- `AZURE_OPENAI_API_KEY`: The API key for accessing Azure OpenAI services.
- `AZURE_OPENAI_ENDPOINT`: The endpoint URL for Azure OpenAI services.
- `OPENAI_API_VERSION`: The version of the OpenAI API to use (default is `2024-08-01-preview`).
- `SEARCH_KEY`: The API key for accessing the Azure Search service.
- `SEARCH_ENDPOINT`: The endpoint URL for the Azure Search service.
- `INDEX_NAME`: The name of the search index to use.
- `LOGGING`: A boolean flag (`True` or `False`) to enable or disable logging.
- `LOG_INDEX`: A boolean flag (`True` or `False`) to enable or disable logging of the search index.
- `OPENAI_ENDPOINT`: The endpoint URL for the local OpenAI service.
- `OPENAI_API_KEY`: The API key for accessing the local OpenAI service.
- `MODEL_NAME`: The name of the AI model to use (e.g., `gpt-4o` or an Ollama models name).
- `USE_AZURE`: A boolean flag (`True` or `False`) to specify whether to use Azure services for the large language model hosting.

## Usage

1. Run the chatbot:
    ```sh
    python gradiocolpalichatbot.py
    ```

2. Open the Gradio interface in your browser to interact with the chatbot.

## Code Overview

### Main Components

- **AzureClient**: Handles communication with Azure AI services.
- **SearchClientWrapper**: Wraps the search client for PDF document search.
- **ModelWrapper**: Wraps the AI model for generating responses.
- **ImageProcessor**: Processes images extracted from PDF documents.
- **PDFIndexer**: Indexes PDF documents and uploads them to the search index.
- **Chatbot**: Main chatbot class that integrates all components and handles user interactions.

### Key Functions

- **log_query_results**: Logs the query results to an HTML file in the [logs](http://_vscodecontentref_/2) directory.
- **main**: Initializes all components and starts the Gradio interface.

## Logging

Query results are logged in the [logs](http://_vscodecontentref_/3) directory with a timestamped HTML file. Each log file contains the query text, top results, and extracted images.

## License

This project is licensed under the MIT License. See the [LICENSE](http://_vscodecontentref_/4) file for details.