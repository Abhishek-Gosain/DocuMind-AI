# DocuMind AI – Local Document Q&A Assistant

DocuMind AI lets you upload a PDF or text document and ask questions about its contents. It finds the most relevant sections of your document and uses a locally running language model to generate answers based on that information.

The project uses **Retrieval-Augmented Generation (RAG)** to make document-based question answering more relevant and useful, without requiring a paid AI API.

## Features

- **PDF and TXT Support:** Upload PDF or plain-text documents.
- **Semantic Search:** Find relevant information based on the meaning of your question.
- **RAG Pipeline:** Retrieve relevant document sections before generating an answer.
- **Local AI Model:** Generate answers using Ollama and Llama 3.2.
- **Text Embeddings:** Convert document sections and questions into numerical representations for semantic search.
- **Simple Interface:** Upload a document, ask a question, and view the answer.
- **Source Transparency:** View the document sections used to generate an answer.
- **No Paid API Required:** Run the application using locally installed models.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | User interface |
| Ollama | Run the language model locally |
| Llama 3.2 (1B) | Generate answers |
| Sentence Transformers | Create text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| NumPy | Similarity calculations |
| PyMuPDF | Extract text from PDF files |
| Requests | Communicate with the local Ollama API |

## How It Works

1. **Upload a document:** The application accepts a PDF or TXT file.
2. **Extract the text:** Text is extracted from the uploaded document.
3. **Split the text into chunks:** The document is divided into smaller, overlapping sections.
4. **Create embeddings:** Each section is converted into a numerical vector using the embedding model.
5. **Ask a question:** Enter a question about the document.
6. **Retrieve relevant sections:** The application compares the question embedding with the document embeddings and selects the most relevant sections.
7. **Generate an answer:** The retrieved information is sent to the local language model, which generates an answer based on the provided context.
8. **Review the sources:** Expand the sources section to see which document sections were used.

## Project Structure

```text
documind-ai/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

The `.venv` virtual environment and downloaded model files are not included in the repository.

## Getting Started

### Prerequisites

Install the following before running the application:

- Python 3.10 or later
- [Ollama](https://ollama.com/)
- Git (optional, for cloning the repository)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/documind-ai.git
cd documind-ai
```

Replace `YOUR-USERNAME` with your GitHub username. If you have already downloaded the project, open a terminal in the project folder instead.

### 2. Create a Virtual Environment

For macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Language Model

Make sure Ollama is installed, then run:

```bash
ollama pull llama3.2:1b
```

This downloads the model to your computer. The first download may take some time.

### 5. Start Ollama

Ensure the Ollama service is running. If it is not already running, open another terminal and run:

```bash
ollama serve
```

If Ollama is already running in the background, you do not need to start it again.

### 6. Run the Application

From the project directory, run:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open the URL in your browser, upload a PDF or TXT file, and start asking questions.

## Configuration

The application uses the following local models:

- **Language model:** `llama3.2:1b`
- **Embedding model:** `all-MiniLM-L6-v2`

The embedding model is downloaded automatically when it is first loaded. An internet connection may be required during the initial model downloads.

After setup, document processing, semantic retrieval, and answer generation run locally on your computer.

## Current Scope

This version focuses on single-document question answering. It uses in-memory text chunks and embeddings, retrieves the top three relevant sections, and passes them to the language model as context.

The application does not currently provide persistent vector storage or a multi-document knowledge base.

## Future Improvements

- Add persistent vector storage with ChromaDB.
- Support multiple document uploads and document collections.
- Add conversation history and follow-up questions.
- Improve source references with page numbers.
- Add document summaries and key-point extraction.
- Improve handling of long documents and retrieval accuracy.

## Learning Outcomes

This project demonstrates practical experience with:

- Retrieval-Augmented Generation (RAG)
- Large Language Model (LLM) integration
- Text embeddings and semantic search
- Document processing and text chunking
- Similarity-based information retrieval
- Local AI model deployment
- Python application development

## Privacy

Documents are processed by the application on your local machine, and answers are generated through the local Ollama service. The application does not require a paid cloud AI API. Initial software and model downloads may require an internet connection.

## License

This project is available for learning and portfolio demonstration. Add a license file if you want to specify formal reuse and distribution permissions.
