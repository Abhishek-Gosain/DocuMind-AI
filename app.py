import requests
import streamlit as st
import pymupdf
import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="centered",
)

st.title("📚 DocuMind AI")
st.subheader("Your AI-powered document assistant")

st.write(
    "Upload a PDF or TXT document and ask questions "
    "to find relevant information."
)


# --------------------------------------------------
# 2. EXTRACT TEXT FROM PDF OR TXT
# --------------------------------------------------

def extract_text(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".txt"):
        text = uploaded_file.getvalue().decode("utf-8-sig")

    elif file_name.endswith(".pdf"):
        pdf_document = pymupdf.open(
            stream=uploaded_file.getvalue(),
            filetype="pdf",
        )

        try:
            pages = []

            for page_number, page in enumerate(
                pdf_document,
                start=1,
            ):
                page_text = page.get_text()
                pages.append(
                    f"Page {page_number}\n{page_text}"
                )

            text = "\n\n".join(pages)

        finally:
            pdf_document.close()

    else:
        raise ValueError(
            "Please upload a PDF or TXT file."
        )

    return text.strip()


# --------------------------------------------------
# 3. SPLIT TEXT INTO CHUNKS
# --------------------------------------------------

def chunk_text(text, chunk_size=500, overlap=100):
    if chunk_size <= 0:
        raise ValueError(
            "Chunk size must be positive."
        )

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be non-negative and "
            "smaller than chunk size."
        )

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunk = text[start:start + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

        start += step

    return chunks


# --------------------------------------------------
# 4. LOAD EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# --------------------------------------------------
# 5. CREATE EMBEDDINGS
# --------------------------------------------------

def create_embeddings(chunks, model):
    return model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


# --------------------------------------------------
# 6. RETRIEVE RELEVANT CHUNKS
# --------------------------------------------------

def retrieve_relevant_chunks(
    question,
    chunks,
    embeddings,
    model,
    top_k=3,
):
    question_embedding = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Embeddings are normalized, so dot product gives
    # cosine similarity.
    similarities = np.dot(
        embeddings,
        question_embedding,
    )

    top_indices = np.argsort(
        similarities
    )[::-1][:min(top_k, len(chunks))]

    return [
        {
            "text": chunks[int(index)],
            "score": float(similarities[index]),
            "index": int(index),
        }
        for index in top_indices
    ]


# --------------------------------------------------
# 7. GENERATE ANSWER USING LOCAL OLLAMA
# --------------------------------------------------

def ask_local_ai(question, context):
    url = "http://localhost:11434/api/chat"

    prompt = f"""
You are DocuMind AI, a document question-answering assistant.

Answer the user's question using only the document
excerpts provided below.

DOCUMENT EXCERPTS:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Answer the question directly and clearly.
2. Use facts supported by the supplied excerpts.
3. Do not invent facts or make unsupported assumptions.
4. If the excerpts do not contain enough information,
   say that you could not find the answer in the
   retrieved document sections.
5. Keep the answer concise.
6. Include all relevant details available in the excerpts.

ANSWER:
"""

    payload = {
        "model": "llama3.2:1b",
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
    }

    response = requests.post(
        url,
        json=payload,
        timeout=180,
    )

    response.raise_for_status()

    result = response.json()

    return result["message"]["content"]


# --------------------------------------------------
# 8. UPLOAD AND PROCESS DOCUMENT
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "txt"],
)

document_text = ""
chunks = []
embeddings = None
model = None

if uploaded_file is not None:
    try:
        document_text = extract_text(uploaded_file)

        if not document_text:
            st.warning(
                "No readable text was found. "
                "Scanned PDFs may require OCR."
            )

        else:
            st.success(
                f"Document uploaded: {uploaded_file.name}"
            )

            chunks = chunk_text(
                document_text,
                chunk_size=500,
                overlap=100,
            )

            with st.spinner(
                "Preparing your document for AI questions..."
            ):
                model = load_embedding_model()

                embeddings = create_embeddings(
                    chunks,
                    model,
                )

            st.caption(
                f"Document ready • {len(chunks)} text chunks indexed"
            )

            # Optional document previews for development.
            # These are hidden from the normal user interface.
            with st.expander("Document details"):
                st.write(
                    "Characters extracted:",
                    len(document_text),
                )

                st.write(
                    "Number of chunks:",
                    len(chunks),
                )

                st.write(
                    "Embedding dimensions:",
                    embeddings.shape[1],
                )

                with st.expander("Preview extracted text"):
                    st.text(document_text[:5000])

                with st.expander("Preview document chunks"):
                    for index, chunk in enumerate(
                        chunks,
                        start=1,
                    ):
                        st.markdown(f"**Chunk {index}**")
                        st.text(chunk)
                        st.divider()

    except Exception as error:
        st.error(
            f"Could not process the document: {error}"
        )

        document_text = ""
        chunks = []
        embeddings = None
        model = None


# --------------------------------------------------
# 9. SINGLE ASK AI INTERFACE
# --------------------------------------------------

st.divider()
st.subheader("💬 Ask AI")

question = st.text_input(
    "Ask a question about your document",
    key="document_question",
    placeholder="Ask questions about the uploaded document",
)

if st.button("Ask AI", type="primary"):
    if uploaded_file is None:
        st.warning(
            "Please upload a PDF or TXT document first."
        )

    elif not question.strip():
        st.warning(
            "Please enter a question."
        )

    elif embeddings is None or model is None or not chunks:
        st.warning(
            "The document could not be processed. "
            "Please upload it again."
        )

    else:
        try:
            # Semantic search happens in the background.
            with st.spinner(
                "Finding relevant information..."
            ):
                results = retrieve_relevant_chunks(
                    question.strip(),
                    chunks,
                    embeddings,
                    model,
                    top_k=3,
                )

                context = "\n\n".join(
                    [
                        f"Document excerpt {rank}:\n"
                        f"{result['text']}"
                        for rank, result in enumerate(
                            results,
                            start=1,
                        )
                    ]
                )

            # The AI answers using the retrieved sections.
            with st.spinner(
                "Generating your answer..."
            ):
                answer = ask_local_ai(
                    question.strip(),
                    context,
                )

            st.markdown("### Answer")
            st.write(answer)

            # Keep source information optional.
            with st.expander(
                "View sources used for this answer"
            ):
                for rank, result in enumerate(
                    results,
                    start=1,
                ):
                    st.markdown(
                        f"**Source {rank}** "
                        f"(chunk {result['index'] + 1})"
                    )

                    st.text(result["text"])

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to Ollama. "
                "Please make sure Ollama is running."
            )

        except requests.exceptions.Timeout:
            st.error(
                "The AI took too long to respond. "
                "Please try again."
            )

        except requests.exceptions.RequestException as error:
            st.error(
                f"Request to Ollama failed: {error}"
            )

        except Exception as error:
            st.error(
                f"Something went wrong: {error}"
            )
