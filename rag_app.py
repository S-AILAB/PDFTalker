# """
# Simple RAG app using Gemini.
# Upload one or more small PDFs, ask a question in the input box, get an answer grounded in them.

# How this works, in plain terms.
# The text from every uploaded PDF is pulled out and cut into small chunks.
# Each chunk is converted into a vector using the Gemini embedding model, and remembers which file it came from.
# When a question comes in, it is converted into a vector too.
# The chunks whose vectors are closest to the question's vector are picked as context, regardless of which file they came from.
# Those chunks, along with the question, are sent to a Gemini text model to produce the final answer.
# Every question and answer is kept in a running history for the session, until you choose to end it.
# """

# import streamlit as st
# from pypdf import PdfReader
# from google import genai
# from google.genai import types
# import numpy as np


# EMBED_MODEL = "gemini-embedding-001"
# CHAT_MODEL = "gemini-3.5-flash"
# CHUNK_SIZE = 800
# CHUNK_OVERLAP = 150
# TOP_K = 4


# def extract_text_from_pdf(file):
#     reader = PdfReader(file)
#     pages = []
#     for page in reader.pages:
#         text = page.extract_text() or ""
#         pages.append(text)
#     return "\n".join(pages)


# def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
#     chunks = []
#     start = 0
#     length = len(text)
#     while start < length:
#         end = min(start + chunk_size, length)
#         chunk = text[start:end].strip()
#         if chunk:
#             chunks.append(chunk)
#         start += chunk_size - overlap
#     return chunks


# def build_index_from_files(client, uploaded_files):
#     """Read every uploaded PDF, chunk it, and embed each chunk.
#     Returns a list of chunks and a matching list of source file names."""
#     all_chunks = []
#     all_sources = []
#     for uploaded_file in uploaded_files:
#         raw_text = extract_text_from_pdf(uploaded_file)
#         if not raw_text.strip():
#             continue
#         chunks = split_into_chunks(raw_text)
#         all_chunks.extend(chunks)
#         all_sources.extend([uploaded_file.name] * len(chunks))
#     if not all_chunks:
#         return [], [], []
#     chunk_vectors = embed_texts(client, all_chunks, task_type="RETRIEVAL_DOCUMENT")
#     return all_chunks, all_sources, chunk_vectors


# def embed_texts(client, texts, task_type):
#     result = client.models.embed_content(
#         model=EMBED_MODEL,
#         contents=texts,
#         config=types.EmbedContentConfig(task_type=task_type),
#     )
#     return [np.array(e.values) for e in result.embeddings]


# def cosine_similarity(a, b):
#     return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# def get_top_chunks(query_vector, chunk_vectors, chunks, sources, top_k=TOP_K):
#     scores = [cosine_similarity(query_vector, vec) for vec in chunk_vectors]
#     ranked = sorted(zip(scores, chunks, sources), key=lambda item: item[0], reverse=True)
#     return [(chunk, source) for score, chunk, source in ranked[:top_k]]


# def build_prompt(question, context_pairs):
#     context_text = "\n\n---\n\n".join(
#         f"From {source}:\n{chunk}" for chunk, source in context_pairs
#     )
#     prompt = (
#         "You are answering a question using only the context below, taken from PDF "
#         "files the user uploaded. If the answer is not in the context, say you cannot "
#         "find it in the documents rather than guessing.\n\n"
#         f"Context:\n{context_text}\n\n"
#         f"Question: {question}\n\n"
#         "Answer clearly and briefly."
#     )
#     return prompt


# def init_session_state():
#     defaults = {
#         "chunks": None,
#         "sources": None,
#         "chunk_vectors": None,
#         "indexed_file_names": None,
#         "history": [],
#     }
#     for key, value in defaults.items():
#         if key not in st.session_state:
#             st.session_state[key] = value


# def end_session():
#     st.session_state.chunks = None
#     st.session_state.sources = None
#     st.session_state.chunk_vectors = None
#     st.session_state.indexed_file_names = None
#     st.session_state.history = []


# def main():
#     st.set_page_config(page_title="PDF Question Answering", page_icon="📄")
#     st.title("Ask questions about your PDFs")
#     st.write("Upload one or more small PDFs, then type a question about them below.")

#     init_session_state()

#     api_key = st.text_input("Gemini API key", type="password")
#     uploaded_files = st.file_uploader(
#         "Upload PDFs", type=["pdf"], accept_multiple_files=True
#     )

#     if not api_key:
#         st.info("Enter your Gemini API key above to get started.")
#         return

#     client = genai.Client(api_key=api_key)

#     current_names = sorted(f.name for f in uploaded_files) if uploaded_files else None
#     if uploaded_files and current_names != st.session_state.indexed_file_names:
#         with st.spinner("Reading and indexing your PDFs..."):
#             chunks, sources, chunk_vectors = build_index_from_files(client, uploaded_files)
#             if not chunks:
#                 st.error("No readable text was found in these PDFs. They may be scanned images.")
#                 return
#             st.session_state.chunks = chunks
#             st.session_state.sources = sources
#             st.session_state.chunk_vectors = chunk_vectors
#             st.session_state.indexed_file_names = current_names
#         file_count = len(uploaded_files)
#         st.success(f"Indexed {len(chunks)} chunks from {file_count} file(s).")

#     if st.session_state.chunks is None:
#         st.info("Upload at least one PDF to continue.")
#         return

#     question = st.text_input("Your question", key="question_box")

#     if question:
#         with st.spinner("Thinking..."):
#             query_vector = embed_texts(client, [question], task_type="RETRIEVAL_QUERY")[0]
#             top_pairs = get_top_chunks(
#                 query_vector,
#                 st.session_state.chunk_vectors,
#                 st.session_state.chunks,
#                 st.session_state.sources,
#             )
#             prompt = build_prompt(question, top_pairs)
#             response = client.models.generate_content(
#                 model=CHAT_MODEL,
#                 contents=prompt,
#             )
#         answer = response.text
#         st.session_state.history.append((question, answer, top_pairs))

#     if st.session_state.history:
#         last_question, last_answer, last_pairs = st.session_state.history[-1]
#         st.markdown("### Answer")
#         st.write(last_answer)

#         with st.expander("Show the passages used to answer this"):
#             for i, (chunk, source) in enumerate(last_pairs, start=1):
#                 st.markdown(f"**Passage {i}, from {source}**")
#                 st.write(chunk)

#         st.markdown("### Conversation history")
#         for past_question, past_answer, _ in reversed(st.session_state.history):
#             st.markdown(f"**Q: {past_question}**")
#             st.write(past_answer)
#             st.markdown("---")

#         if st.button("End session"):
#             end_session()
#             st.rerun()


# if __name__ == "__main__":
#     main()



# """
# Simple RAG app using Gemini.
# Upload one or more small PDFs, ask a question in the input box, get an answer grounded in them.

# How this works, in plain terms.
# The text from every uploaded PDF is pulled out and cut into small chunks.
# Each chunk is converted into a vector using the Gemini embedding model, and remembers which file it came from.
# When a question comes in, it is converted into a vector too.
# The chunks whose vectors are closest to the question's vector are picked as context, regardless of which file they came from.
# Those chunks, along with the question, are sent to a Gemini text model to produce the final answer.
# Every question and answer is kept in a running history for the session, until you choose to end it.
# """

# import streamlit as st
# from pypdf import PdfReader
# from google import genai
# from google.genai import types
# import numpy as np


# EMBED_MODEL = "gemini-embedding-001"
# CHAT_MODEL = "gemini-3.5-flash"
# CHUNK_SIZE = 800
# CHUNK_OVERLAP = 150
# TOP_K = 4


# def extract_text_from_pdf(file):
#     reader = PdfReader(file)
#     pages = []
#     for page in reader.pages:
#         text = page.extract_text() or ""
#         pages.append(text)
#     return "\n".join(pages)


# def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
#     chunks = []
#     start = 0
#     length = len(text)
#     while start < length:
#         end = min(start + chunk_size, length)
#         chunk = text[start:end].strip()
#         if chunk:
#             chunks.append(chunk)
#         start += chunk_size - overlap
#     return chunks


# def build_index_from_files(client, uploaded_files):
#     """Read every uploaded PDF, chunk it, and embed each chunk.
#     Returns a list of chunks and a matching list of source file names."""
#     all_chunks = []
#     all_sources = []
#     for uploaded_file in uploaded_files:
#         raw_text = extract_text_from_pdf(uploaded_file)
#         if not raw_text.strip():
#             continue
#         chunks = split_into_chunks(raw_text)
#         all_chunks.extend(chunks)
#         all_sources.extend([uploaded_file.name] * len(chunks))
#     if not all_chunks:
#         return [], [], []
#     chunk_vectors = embed_texts(client, all_chunks, task_type="RETRIEVAL_DOCUMENT")
#     return all_chunks, all_sources, chunk_vectors


# def embed_texts(client, texts, task_type):
#     result = client.models.embed_content(
#         model=EMBED_MODEL,
#         contents=texts,
#         config=types.EmbedContentConfig(task_type=task_type),
#     )
#     return [np.array(e.values) for e in result.embeddings]


# def cosine_similarity(a, b):
#     return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# def get_top_chunks(query_vector, chunk_vectors, chunks, sources, top_k=TOP_K):
#     scores = [cosine_similarity(query_vector, vec) for vec in chunk_vectors]
#     ranked = sorted(zip(scores, chunks, sources), key=lambda item: item[0], reverse=True)
#     return [(chunk, source) for score, chunk, source in ranked[:top_k]]


# def build_prompt(question, context_pairs):
#     context_text = "\n\n---\n\n".join(
#         f"From {source}:\n{chunk}" for chunk, source in context_pairs
#     )
#     prompt = (
#         "You are answering a question using only the context below, taken from PDF "
#         "files the user uploaded. If the answer is not in the context, say you cannot "
#         "find it in the documents rather than guessing.\n\n"
#         f"Context:\n{context_text}\n\n"
#         f"Question: {question}\n\n"
#         "Answer clearly and briefly."
#     )
#     return prompt


# def init_session_state():
#     defaults = {
#         "chunks": None,
#         "sources": None,
#         "chunk_vectors": None,
#         "indexed_file_names": None,
#         "history": [],
#     }
#     for key, value in defaults.items():
#         if key not in st.session_state:
#             st.session_state[key] = value


# def end_session():
#     st.session_state.chunks = None
#     st.session_state.sources = None
#     st.session_state.chunk_vectors = None
#     st.session_state.indexed_file_names = None
#     st.session_state.history = []


# def main():
#     st.set_page_config(page_title="PDF Question Answering", page_icon="📄")
#     st.title("Ask questions about your PDFs")
#     st.write("Upload one or more small PDFs, then type a question about them below.")

#     init_session_state()

#     api_key = st.secrets.get("GEMINI_API_KEY")
#     uploaded_files = st.file_uploader(
#         "Upload PDFs", type=["pdf"], accept_multiple_files=True
#     )

#     if not api_key:
#         st.error("No Gemini API key found. Add GEMINI_API_KEY in your app's Secrets.")
#         return

#     client = genai.Client(api_key=api_key)

#     current_names = sorted(f.name for f in uploaded_files) if uploaded_files else None
#     if uploaded_files and current_names != st.session_state.indexed_file_names:
#         with st.spinner("Reading and indexing your PDFs..."):
#             chunks, sources, chunk_vectors = build_index_from_files(client, uploaded_files)
#             if not chunks:
#                 st.error("No readable text was found in these PDFs. They may be scanned images.")
#                 return
#             st.session_state.chunks = chunks
#             st.session_state.sources = sources
#             st.session_state.chunk_vectors = chunk_vectors
#             st.session_state.indexed_file_names = current_names
#         file_count = len(uploaded_files)
#         st.success(f"Indexed {len(chunks)} chunks from {file_count} file(s).")

#     if st.session_state.chunks is None:
#         st.info("Upload at least one PDF to continue.")
#         return

#     question = st.text_input("Your question", key="question_box")

#     if question:
#         with st.spinner("Thinking..."):
#             query_vector = embed_texts(client, [question], task_type="RETRIEVAL_QUERY")[0]
#             top_pairs = get_top_chunks(
#                 query_vector,
#                 st.session_state.chunk_vectors,
#                 st.session_state.chunks,
#                 st.session_state.sources,
#             )
#             prompt = build_prompt(question, top_pairs)
#             response = client.models.generate_content(
#                 model=CHAT_MODEL,
#                 contents=prompt,
#             )
#         answer = response.text
#         st.session_state.history.append((question, answer, top_pairs))

#     if st.session_state.history:
#         last_question, last_answer, last_pairs = st.session_state.history[-1]
#         st.markdown("### Answer")
#         st.write(last_answer)

#         with st.expander("Show the passages used to answer this"):
#             for i, (chunk, source) in enumerate(last_pairs, start=1):
#                 st.markdown(f"**Passage {i}, from {source}**")
#                 st.write(chunk)

#         st.markdown("### Conversation history")
#         for past_question, past_answer, _ in reversed(st.session_state.history):
#             st.markdown(f"**Q: {past_question}**")
#             st.write(past_answer)
#             st.markdown("---")

#         if st.button("End session"):
#             end_session()
#             st.rerun()


# if __name__ == "__main__":
#     main()



# """
# Simple RAG app using Gemini.
# Upload one or more small PDFs, ask a question in the input box, get an answer grounded in them.

# How this works, in plain terms.
# The text from every uploaded PDF is pulled out and cut into small chunks.
# Each chunk is converted into a vector using the Gemini embedding model, and remembers which file it came from.
# When a question comes in, it is converted into a vector too.
# The chunks whose vectors are closest to the question's vector are picked as context, regardless of which file they came from.
# Those chunks, along with the question, are sent to a Gemini text model to produce the final answer.
# Every question and answer is kept in a running history for the session, until you choose to end it.
# """

# import streamlit as st
# from pypdf import PdfReader
# from google import genai
# from google.genai import types
# import numpy as np


# EMBED_MODEL = "gemini-embedding-001"
# CHAT_MODEL = "gemini-3.5-flash"
# CHUNK_SIZE = 800
# CHUNK_OVERLAP = 150
# TOP_K = 4


# def extract_text_from_pdf(file):
#     reader = PdfReader(file)
#     pages = []
#     for page in reader.pages:
#         text = page.extract_text() or ""
#         pages.append(text)
#     return "\n".join(pages)


# def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
#     chunks = []
#     start = 0
#     length = len(text)
#     while start < length:
#         end = min(start + chunk_size, length)
#         chunk = text[start:end].strip()
#         if chunk:
#             chunks.append(chunk)
#         start += chunk_size - overlap
#     return chunks


# def build_index_from_files(client, uploaded_files):
#     """Read every uploaded PDF, chunk it, and embed each chunk.
#     Returns a list of chunks and a matching list of source file names."""
#     all_chunks = []
#     all_sources = []
#     for uploaded_file in uploaded_files:
#         raw_text = extract_text_from_pdf(uploaded_file)
#         if not raw_text.strip():
#             continue
#         chunks = split_into_chunks(raw_text)
#         all_chunks.extend(chunks)
#         all_sources.extend([uploaded_file.name] * len(chunks))
#     if not all_chunks:
#         return [], [], []
#     chunk_vectors = embed_texts(client, all_chunks, task_type="RETRIEVAL_DOCUMENT")
#     return all_chunks, all_sources, chunk_vectors


# def embed_texts(client, texts, task_type):
#     result = client.models.embed_content(
#         model=EMBED_MODEL,
#         contents=texts,
#         config=types.EmbedContentConfig(task_type=task_type),
#     )
#     return [np.array(e.values) for e in result.embeddings]


# def cosine_similarity(a, b):
#     return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# def get_top_chunks(query_vector, chunk_vectors, chunks, sources, top_k=TOP_K):
#     scores = [cosine_similarity(query_vector, vec) for vec in chunk_vectors]
#     ranked = sorted(zip(scores, chunks, sources), key=lambda item: item[0], reverse=True)
#     return [(chunk, source) for score, chunk, source in ranked[:top_k]]


# def build_prompt(question, context_pairs):
#     context_text = "\n\n---\n\n".join(
#         f"From {source}:\n{chunk}" for chunk, source in context_pairs
#     )
#     prompt = (
#         "You are answering a question using only the context below, taken from PDF "
#         "files the user uploaded. If the answer is not in the context, say you cannot "
#         "find it in the documents rather than guessing.\n\n"
#         f"Context:\n{context_text}\n\n"
#         f"Question: {question}\n\n"
#         "Answer clearly and briefly."
#     )
#     return prompt


# def init_session_state():
#     defaults = {
#         "chunks": None,
#         "sources": None,
#         "chunk_vectors": None,
#         "indexed_file_names": None,
#         "history": [],
#     }
#     for key, value in defaults.items():
#         if key not in st.session_state:
#             st.session_state[key] = value


# def end_session():
#     st.session_state.chunks = None
#     st.session_state.sources = None
#     st.session_state.chunk_vectors = None
#     st.session_state.indexed_file_names = None
#     st.session_state.history = []


# def main():
#     st.set_page_config(page_title="PDF Question Answering", page_icon="📄")

#     with st.sidebar:
#         st.markdown("## 📄 PDFTalker")
#         st.caption("SID AI LAB")

#     st.title("Ask questions about your PDFs")
#     st.write("Upload one or more small PDFs, then type a question about them below.")

#     init_session_state()

#     api_key = st.secrets.get("GEMINI_API_KEY")
#     uploaded_files = st.file_uploader(
#         "Upload PDFs", type=["pdf"], accept_multiple_files=True
#     )

#     if not api_key:
#         st.error("No Gemini API key found. Add GEMINI_API_KEY in your app's Secrets.")
#         return

#     client = genai.Client(api_key=api_key)

#     current_names = sorted(f.name for f in uploaded_files) if uploaded_files else None
#     if uploaded_files and current_names != st.session_state.indexed_file_names:
#         with st.spinner("Reading and indexing your PDFs..."):
#             chunks, sources, chunk_vectors = build_index_from_files(client, uploaded_files)
#             if not chunks:
#                 st.error("No readable text was found in these PDFs. They may be scanned images.")
#                 return
#             st.session_state.chunks = chunks
#             st.session_state.sources = sources
#             st.session_state.chunk_vectors = chunk_vectors
#             st.session_state.indexed_file_names = current_names
#         file_count = len(uploaded_files)
#         st.success(f"Indexed {len(chunks)} chunks from {file_count} file(s).")

#     if st.session_state.chunks is None:
#         st.info("Upload at least one PDF to continue.")
#         return

#     question = st.text_input("Your question", key="question_box")

#     if question:
#         with st.spinner("Thinking..."):
#             query_vector = embed_texts(client, [question], task_type="RETRIEVAL_QUERY")[0]
#             top_pairs = get_top_chunks(
#                 query_vector,
#                 st.session_state.chunk_vectors,
#                 st.session_state.chunks,
#                 st.session_state.sources,
#             )
#             prompt = build_prompt(question, top_pairs)
#             response = client.models.generate_content(
#                 model=CHAT_MODEL,
#                 contents=prompt,
#             )
#         answer = response.text
#         st.session_state.history.append((question, answer, top_pairs))

#     if st.session_state.history:
#         last_question, last_answer, last_pairs = st.session_state.history[-1]
#         st.markdown("### Answer")
#         st.write(last_answer)

#         with st.expander("Show the passages used to answer this"):
#             for i, (chunk, source) in enumerate(last_pairs, start=1):
#                 st.markdown(f"**Passage {i}, from {source}**")
#                 st.write(chunk)

#         st.markdown("### Conversation history")
#         for past_question, past_answer, _ in reversed(st.session_state.history):
#             st.markdown(f"**Q: {past_question}**")
#             st.write(past_answer)
#             st.markdown("---")

#         if st.button("End session"):
#             end_session()
#             st.rerun()


# if __name__ == "__main__":
#     main()




"""
Simple RAG app using Gemini.
Upload one or more small PDFs, ask a question in the input box, get an answer grounded in them.

How this works, in plain terms.
The text from every uploaded PDF is pulled out and cut into small chunks.
Each chunk is converted into a vector using the Gemini embedding model, and remembers which file it came from.
When a question comes in, it is converted into a vector too.
The chunks whose vectors are closest to the question's vector are picked as context, regardless of which file they came from.
Those chunks, along with the question, are sent to a Gemini text model to produce the final answer.
Every question and answer is kept in a running history for the session, until you choose to end it.
"""

import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types
import numpy as np


EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4


def extract_text_from_pdf(file):
    reader = PdfReader(file)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_index_from_files(client, uploaded_files):
    """Read every uploaded PDF, chunk it, and embed each chunk.
    Returns a list of chunks and a matching list of source file names."""
    all_chunks = []
    all_sources = []
    for uploaded_file in uploaded_files:
        raw_text = extract_text_from_pdf(uploaded_file)
        if not raw_text.strip():
            continue
        chunks = split_into_chunks(raw_text)
        all_chunks.extend(chunks)
        all_sources.extend([uploaded_file.name] * len(chunks))
    if not all_chunks:
        return [], [], []
    chunk_vectors = embed_texts(client, all_chunks, task_type="RETRIEVAL_DOCUMENT")
    return all_chunks, all_sources, chunk_vectors


def embed_texts(client, texts, task_type):
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [np.array(e.values) for e in result.embeddings]


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def get_top_chunks(query_vector, chunk_vectors, chunks, sources, top_k=TOP_K):
    scores = [cosine_similarity(query_vector, vec) for vec in chunk_vectors]
    ranked = sorted(zip(scores, chunks, sources), key=lambda item: item[0], reverse=True)
    return [(chunk, source) for score, chunk, source in ranked[:top_k]]


def build_prompt(question, context_pairs):
    context_text = "\n\n---\n\n".join(
        f"From {source}:\n{chunk}" for chunk, source in context_pairs
    )
    prompt = (
        "You are answering a question using only the context below, taken from PDF "
        "files the user uploaded. If the answer is not in the context, say you cannot "
        "find it in the documents rather than guessing.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer clearly and briefly."
    )
    return prompt


def init_session_state():
    defaults = {
        "chunks": None,
        "sources": None,
        "chunk_vectors": None,
        "indexed_file_names": None,
        "history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def end_session():
    st.session_state.chunks = None
    st.session_state.sources = None
    st.session_state.chunk_vectors = None
    st.session_state.indexed_file_names = None
    st.session_state.history = []


def main():
    st.set_page_config(page_title="PDF Question Answering", page_icon="📄")

    with st.sidebar:
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #4F46E5, #7C3AED);
                padding: 18px 16px;
                border-radius: 12px;
                margin-bottom: 12px;
            ">
                <div style="font-size: 28px; line-height: 1;">📄</div>
                <div style="color: white; font-size: 20px; font-weight: 700; margin-top: 6px;">
                    PDFTalker
                </div>
                <div style="color: rgba(255,255,255,0.75); font-size: 12px; letter-spacing: 1px; margin-top: 2px;">
                    SID AI LAB
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.title("Ask questions about your PDFs")
    st.write("Upload one or more small PDFs, then type a question about them below.")

    init_session_state()

    api_key = st.secrets.get("GEMINI_API_KEY")
    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True
    )

    if not api_key:
        st.error("No Gemini API key found. Add GEMINI_API_KEY in your app's Secrets.")
        return

    client = genai.Client(api_key=api_key)

    current_names = sorted(f.name for f in uploaded_files) if uploaded_files else None
    if uploaded_files and current_names != st.session_state.indexed_file_names:
        with st.spinner("Reading and indexing your PDFs..."):
            chunks, sources, chunk_vectors = build_index_from_files(client, uploaded_files)
            if not chunks:
                st.error("No readable text was found in these PDFs. They may be scanned images.")
                return
            st.session_state.chunks = chunks
            st.session_state.sources = sources
            st.session_state.chunk_vectors = chunk_vectors
            st.session_state.indexed_file_names = current_names
        file_count = len(uploaded_files)
        st.success(f"Indexed {len(chunks)} chunks from {file_count} file(s).")

    if st.session_state.chunks is None:
        st.info("Upload at least one PDF to continue.")
        return

    question = st.text_input("Your question", key="question_box")

    if question:
        with st.spinner("Thinking..."):
            query_vector = embed_texts(client, [question], task_type="RETRIEVAL_QUERY")[0]
            top_pairs = get_top_chunks(
                query_vector,
                st.session_state.chunk_vectors,
                st.session_state.chunks,
                st.session_state.sources,
            )
            prompt = build_prompt(question, top_pairs)
            response = client.models.generate_content(
                model=CHAT_MODEL,
                contents=prompt,
            )
        answer = response.text
        st.session_state.history.append((question, answer, top_pairs))

    if st.session_state.history:
        last_question, last_answer, last_pairs = st.session_state.history[-1]
        st.markdown("### Answer")
        st.write(last_answer)

        with st.expander("Show the passages used to answer this"):
            for i, (chunk, source) in enumerate(last_pairs, start=1):
                st.markdown(f"**Passage {i}, from {source}**")
                st.write(chunk)

        st.markdown("### Conversation history")
        for past_question, past_answer, _ in reversed(st.session_state.history):
            st.markdown(f"**Q: {past_question}**")
            st.write(past_answer)
            st.markdown("---")

        if st.button("End session"):
            end_session()
            st.rerun()


if __name__ == "__main__":
    main()
