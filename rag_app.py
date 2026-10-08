# # # # """
# # # # Simple RAG app using Gemini.
# # # # Upload one or more small PDFs, ask a question in the input box, get an answer grounded in them.

# # # # How this works, in plain terms.
# # # # The text from every uploaded PDF is pulled out and cut into small chunks.
# # # # Each chunk is converted into a vector using the Gemini embedding model, and remembers which file it came from.
# # # # When a question comes in, it is converted into a vector too.
# # # # The chunks whose vectors are closest to the question's vector are picked as context, regardless of which file they came from.
# # # # Those chunks, along with the question, are sent to a Gemini text model to produce the final answer.
# # # # Every question and answer is kept in a running history for the session, until you choose to end it.
# # # # """

# # # # import streamlit as st
# # # # from pypdf import PdfReader
# # # # from google import genai
# # # # from google.genai import types
# # # # import numpy as np


# # # # EMBED_MODEL = "gemini-embedding-001"
# # # # CHAT_MODEL = "gemini-3.5-flash"
# # # # CHUNK_SIZE = 800
# # # # CHUNK_OVERLAP = 150
# # # # TOP_K = 4


# # # # def extract_text_from_pdf(file):
# # # #     reader = PdfReader(file)
# # # #     pages = []
# # # #     for page in reader.pages:
# # # #         text = page.extract_text() or ""
# # # #         pages.append(text)
# # # #     return "\n".join(pages)


# # # # def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
# # # #     chunks = []
# # # #     start = 0
# # # #     length = len(text)
# # # #     while start < length:
# # # #         end = min(start + chunk_size, length)
# # # #         chunk = text[start:end].strip()
# # # #         if chunk:
# # # #             chunks.append(chunk)
# # # #         start += chunk_size - overlap
# # # #     return chunks


# # # # def build_index_from_files(client, uploaded_files):
# # # #     """Read every uploaded PDF, chunk it, and embed each chunk.
# # # #     Returns a list of chunks and a matching list of source file names."""
# # # #     all_chunks = []
# # # #     all_sources = []
# # # #     for uploaded_file in uploaded_files:
# # # #         raw_text = extract_text_from_pdf(uploaded_file)
# # # #         if not raw_text.strip():
# # # #             continue
# # # #         chunks = split_into_chunks(raw_text)
# # # #         all_chunks.extend(chunks)
# # # #         all_sources.extend([uploaded_file.name] * len(chunks))
# # # #     if not all_chunks:
# # # #         return [], [], []
# # # #     chunk_vectors = embed_texts(client, all_chunks, task_type="RETRIEVAL_DOCUMENT")
# # # #     return all_chunks, all_sources, chunk_vectors


# # # # def embed_texts(client, texts, task_type):
# # # #     result = client.models.embed_content(
# # # #         model=EMBED_MODEL,
# # # #         contents=texts,
# # # #         config=types.EmbedContentConfig(task_type=task_type),
# # # #     )
# # # #     return [np.array(e.values) for e in result.embeddings]


# # # # def cosine_similarity(a, b):
# # # #     return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# # # # def get_top_chunks(query_vector, chunk_vectors, chunks, sources, top_k=TOP_K):
# # # #     scores = [cosine_similarity(query_vector, vec) for vec in chunk_vectors]
# # # #     ranked = sorted(zip(scores, chunks, sources), key=lambda item: item[0], reverse=True)
# # # #     return [(chunk, source) for score, chunk, source in ranked[:top_k]]


# # # # def build_prompt(question, context_pairs):
# # # #     context_text = "\n\n---\n\n".join(
# # # #         f"From {source}:\n{chunk}" for chunk, source in context_pairs
# # # #     )
# # # #     prompt = (
# # # #         "You are answering a question using only the context below, taken from PDF "
# # # #         "files the user uploaded. If the answer is not in the context, say you cannot "
# # # #         "find it in the documents rather than guessing.\n\n"
# # # #         f"Context:\n{context_text}\n\n"
# # # #         f"Question: {question}\n\n"
# # # #         "Answer clearly and briefly."
# # # #     )
# # # #     return prompt


# # # # def init_session_state():
# # # #     defaults = {
# # # #         "chunks": None,
# # # #         "sources": None,
# # # #         "chunk_vectors": None,
# # # #         "indexed_file_names": None,
# # # #         "history": [],
# # # #     }
# # # #     for key, value in defaults.items():
# # # #         if key not in st.session_state:
# # # #             st.session_state[key] = value


# # # # def end_session():
# # # #     st.session_state.chunks = None
# # # #     st.session_state.sources = None
# # # #     st.session_state.chunk_vectors = None
# # # #     st.session_state.indexed_file_names = None
# # # #     st.session_state.history = []


# # # # def main():
# # # #     st.set_page_config(page_title="PDF Question Answering", page_icon="📄")
# # # #     st.title("Ask questions about your PDFs")
# # # #     st.write("Upload one or more small PDFs, then type a question about them below.")

# # # #     init_session_state()

# # # #     api_key = st.text_input("Gemini API key", type="password")
# # # #     uploaded_files = st.file_uploader(
# # # #         "Upload PDFs", type=["pdf"], accept_multiple_files=True
# # # #     )

# # # #     if not api_key:
# # # #         st.info("Enter your Gemini API key above to get started.")
# # # #         return

# # # #     client = genai.Client(api_key=api_key)

# # # #     current_names = sorted(f.name for f in uploaded_files) if uploaded_files else None
# # # #     if uploaded_files and current_names != st.session_state.indexed_file_names:
# # # #         with st.spinner("Reading and indexing your PDFs..."):
# # # #             chunks, sources, chunk_vectors = build_index_from_files(client, uploaded_files)
# # # #             if not chunks:
# # # #                 st.error("No readable text was found in these PDFs. They may be scanned images.")
# # # #                 return
# # # #             st.session_state.chunks = chunks
# # # #             st.session_state.sources = sources
# # # #             st.session_state.chunk_vectors = chunk_vectors
# # # #             st.session_state.indexed_file_names = current_names
# # # #         file_count = len(uploaded_files)
# # # #         st.success(f"Indexed {len(chunks)} chunks from {file_count} file(s).")

# # # #     if st.session_state.chunks is None:
# # # #         st.info("Upload at least one PDF to continue.")
# # # #         return

# # # #     question = st.text_input("Your question", key="question_box")

# # # #     if question:
# # # #         with st.spinner("Thinking..."):
# # # #             query_vector = embed_texts(client, [question], task_type="RETRIEVAL_QUERY")[0]
# # # #             top_pairs = get_top_chunks(
# # # #                 query_vector,
# # # #                 st.session_state.chunk_vectors,
# # # #                 st.session_state.chunks,
# # # #                 st.session_state.sources,
# # # #             )
# # # #             prompt = build_prompt(question, top_pairs)
# # # #             response = client.models.generate_content(
# # # #                 model=CHAT_MODEL,
# # # #                 contents=prompt,
# # # #             )
# # # #         answer = response.text
# # # #         st.session_state.history.append((question, answer, top_pairs))

# # # #     if st.session_state.history:
# # # #         last_question, last_answer, last_pairs = st.session_state.history[-1]
# # # #         st.markdown("### Answer")
# # # #         st.write(last_answer)

# # # #         with st.expander("Show the passages used to answer this"):
# # # #             for i, (chunk, source) in enumerate(last_pairs, start=1):
# # # #                 st.markdown(f"**Passage {i}, from {source}**")
# # # #                 st.write(chunk)

# # # #         st.markdown("### Conversation history")
# # # #         for past_question, past_answer, _ in reversed(st.session_state.history):
# # # #             st.markdown(f"**Q: {past_question}**")
# # # #             st.write(past_answer)
# # # #             st.markdown("---")

# # # #         if st.button("End session"):
# # # #             end_session()
# # # #             st.rerun()


# # # # if __name__ == "__main__":
# # # #     main()



# # # # """
# # # # Simple RAG app using Gemini.
# # # # Upload one or more small PDFs, ask a question in the input box, get an answer grounded in them.

# # # # How this works, in plain terms.
# # # # The text from every uploaded PDF is pulled out and cut into small chunks.
# # # # Each chunk is converted into a vector using the Gemini embedding model, and remembers which file it came from.
# # # # When a question comes in, it is converted into a vector too.
# # # # The chunks whose vectors are closest to the question's vector are picked as context, regardless of which file they came from.
# # # # Those chunks, along with the question, are sent to a Gemini text model to produce the final answer.
# # # # Every question and answer is kept in a running history for the session, until you choose to end it.
# # # # """

# # # # import streamlit as st
# # # # from pypdf import PdfReader
# # # # from google import genai
# # # # from google.genai import types
# # # # import numpy as np


# # # # EMBED_MODEL = "gemini-embedding-001"
# # # # CHAT_MODEL = "gemini-3.5-flash"
# # # # CHUNK_SIZE = 800
# # # # CHUNK_OVERLAP = 150
# # # # TOP_K = 4


# # # # def extract_text_from_pdf(file):
# # # #     reader = PdfReader(file)
# # # #     pages = []
# # # #     for page in reader.pages:
# # # #         text = page.extract_text() or ""
# # # #         pages.append(text)
# # # #     return "\n".join(pages)


# # # # def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
# # # #     chunks = []
# # # #     start = 0
# # # #     length = len(text)
# # # #     while start < length:
# # # #         end = min(start + chunk_size, length)
# # # #         chunk = text[start:end].strip()
# # # #         if chunk:
# # # #             chunks.append(chunk)
# # # #         start += chunk_size - overlap
# # # #     return chunks


# # # # def build_index_from_files(client, uploaded_files):
# # # #     """Read every uploaded PDF, chunk it, and embed each chunk.
# # # #     Returns a list of chunks and a matching list of source file names."""
# # # #     all_chunks = []
# # # #     all_sources = []
# # # #     for uploaded_file in uploaded_files:
# # # #         raw_text = extract_text_from_pdf(uploaded_file)
# # # #         if not raw_text.strip():
# # # #             continue
# # # #         chunks = split_into_chunks(raw_text)
# # # #         all_chunks.extend(chunks)
# # # #         all_sources.extend([uploaded_file.name] * len(chunks))
# # # #     if not all_chunks:
# # # #         return [], [], []
# # # #     chunk_vectors = embed_texts(client, all_chunks, task_type="RETRIEVAL_DOCUMENT")
# # # #     return all_chunks, all_sources, chunk_vectors


# # # # def embed_texts(client, texts, task_type):
# # # #     result = client.models.embed_content(
# # # #         model=EMBED_MODEL,
# # # #         contents=texts,
# # # #         config=types.EmbedContentConfig(task_type=task_type),
# # # #     )
# # # #     return [np.array(e.values) for e in result.embeddings]


# # # # def cosine_similarity(a, b):
# # # #     return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# # # # def get_top_chunks(query_vector, chunk_vectors, chunks, sources, top_k=TOP_K):
# # # #     scores = [cosine_similarity(query_vector, vec) for vec in chunk_vectors]
# # # #     ranked = sorted(zip(scores, chunks, sources), key=lambda item: item[0], reverse=True)
# # # #     return [(chunk, source) for score, chunk, source in ranked[:top_k]]


# # # # def build_prompt(question, context_pairs):
# # # #     context_text = "\n\n---\n\n".join(
# # # #         f"From {source}:\n{chunk}" for chunk, source in context_pairs
# # # #     )
# # # #     prompt = (
# # # #         "You are answering a question using only the context below, taken from PDF "
# # # #         "files the user uploaded. If the answer is not in the context, say you cannot "
# # # #         "find it in the documents rather than guessing.\n\n"
# # # #         f"Context:\n{context_text}\n\n"
# # # #         f"Question: {question}\n\n"
# # # #         "Answer clearly and briefly."
# # # #     )
# # # #     return prompt


# # # # def init_session_state():
# # # #     defaults = {
# # # #         "chunks": None,
# # # #         "sources": None,
# # # #         "chunk_vectors": None,
# # # #         "indexed_file_names": None,
# # # #         "history": [],
# # # #     }
# # # #     for key, value in defaults.items():
# # # #         if key not in st.session_state:
# # # #             st.session_state[key] = value


# # # # def end_session():
# # # #     st.session_state.chunks = None
# # # #     st.session_state.sources = None
# # # #     st.session_state.chunk_vectors = None
# # # #     st.session_state.indexed_file_names = None
# # # #     st.session_state.history = []


# # # # def main():
# # # #     st.set_page_config(page_title="PDF Question Answering", page_icon="📄")
# # # #     st.title("Ask questions about your PDFs")
# # # #     st.write("Upload one or more small PDFs, then type a question about them below.")

# # # #     init_session_state()

# # # #     api_key = st.secrets.get("GEMINI_API_KEY")
# # # #     uploaded_files = st.file_uploader(
# # # #         "Upload PDFs", type=["pdf"], accept_multiple_files=True
# # # #     )

# # # #     if not api_key:
# # # #         st.error("No Gemini API key found. Add GEMINI_API_KEY in your app's Secrets.")
# # # #         return

# # # #     client = genai.Client(api_key=api_key)

# # # #     current_names = sorted(f.name for f in uploaded_files) if uploaded_files else None
# # # #     if uploaded_files and current_names != st.session_state.indexed_file_names:
# # # #         with st.spinner("Reading and indexing your PDFs..."):
# # # #             chunks, sources, chunk_vectors = build_index_from_files(client, uploaded_files)
# # # #             if not chunks:
# # # #                 st.error("No readable text was found in these PDFs. They may be scanned images.")
# # # #                 return
# # # #             st.session_state.chunks = chunks
# # # #             st.session_state.sources = sources
# # # #             st.session_state.chunk_vectors = chunk_vectors
# # # #             st.session_state.indexed_file_names = current_names
# # # #         file_count = len(uploaded_files)
# # # #         st.success(f"Indexed {len(chunks)} chunks from {file_count} file(s).")

# # # #     if st.session_state.chunks is None:
# # # #         st.info("Upload at least one PDF to continue.")
# # # #         return

# # # #     question = st.text_input("Your question", key="question_box")

# # # #     if question:
# # # #         with st.spinner("Thinking..."):
# # # #             query_vector = embed_texts(client, [question], task_type="RETRIEVAL_QUERY")[0]
# # # #             top_pairs = get_top_chunks(
# # # #                 query_vector,
# # # #                 st.session_state.chunk_vectors,
# # # #                 st.session_state.chunks,
# # # #                 st.session_state.sources,
# # # #             )
# # # #             prompt = build_prompt(question, top_pairs)
# # # #             response = client.models.generate_content(
# # # #                 model=CHAT_MODEL,
# # # #                 contents=prompt,
# # # #             )
# # # #         answer = response.text
# # # #         st.session_state.history.append((question, answer, top_pairs))

# # # #     if st.session_state.history:
# # # #         last_question, last_answer, last_pairs = st.session_state.history[-1]
# # # #         st.markdown("### Answer")
# # # #         st.write(last_answer)

# # # #         with st.expander("Show the passages used to answer this"):
# # # #             for i, (chunk, source) in enumerate(last_pairs, start=1):
# # # #                 st.markdown(f"**Passage {i}, from {source}**")
# # # #                 st.write(chunk)

# # # #         st.markdown("### Conversation history")
# # # #         for past_question, past_answer, _ in reversed(st.session_state.history):
# # # #             st.markdown(f"**Q: {past_question}**")
# # # #             st.write(past_answer)
# # # #             st.markdown("---")

# # # #         if st.button("End session"):
# # # #             end_session()
# # # #             st.rerun()


# # # # if __name__ == "__main__":
# # # #     main()



# # # # """
# # # # Simple RAG app using Gemini.
# # # # Upload one or more small PDFs, ask a question in the input box, get an answer grounded in them.

# # # # How this works, in plain terms.
# # # # The text from every uploaded PDF is pulled out and cut into small chunks.
# # # # Each chunk is converted into a vector using the Gemini embedding model, and remembers which file it came from.
# # # # When a question comes in, it is converted into a vector too.
# # # # The chunks whose vectors are closest to the question's vector are picked as context, regardless of which file they came from.
# # # # Those chunks, along with the question, are sent to a Gemini text model to produce the final answer.
# # # # Every question and answer is kept in a running history for the session, until you choose to end it.
# # # # """

# # # # import streamlit as st
# # # # from pypdf import PdfReader
# # # # from google import genai
# # # # from google.genai import types
# # # # import numpy as np


# # # # EMBED_MODEL = "gemini-embedding-001"
# # # # CHAT_MODEL = "gemini-3.5-flash"
# # # # CHUNK_SIZE = 800
# # # # CHUNK_OVERLAP = 150
# # # # TOP_K = 4


# # # # def extract_text_from_pdf(file):
# # # #     reader = PdfReader(file)
# # # #     pages = []
# # # #     for page in reader.pages:
# # # #         text = page.extract_text() or ""
# # # #         pages.append(text)
# # # #     return "\n".join(pages)


# # # # def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
# # # #     chunks = []
# # # #     start = 0
# # # #     length = len(text)
# # # #     while start < length:
# # # #         end = min(start + chunk_size, length)
# # # #         chunk = text[start:end].strip()
# # # #         if chunk:
# # # #             chunks.append(chunk)
# # # #         start += chunk_size - overlap
# # # #     return chunks


# # # # def build_index_from_files(client, uploaded_files):
# # # #     """Read every uploaded PDF, chunk it, and embed each chunk.
# # # #     Returns a list of chunks and a matching list of source file names."""
# # # #     all_chunks = []
# # # #     all_sources = []
# # # #     for uploaded_file in uploaded_files:
# # # #         raw_text = extract_text_from_pdf(uploaded_file)
# # # #         if not raw_text.strip():
# # # #             continue
# # # #         chunks = split_into_chunks(raw_text)
# # # #         all_chunks.extend(chunks)
# # # #         all_sources.extend([uploaded_file.name] * len(chunks))
# # # #     if not all_chunks:
# # # #         return [], [], []
# # # #     chunk_vectors = embed_texts(client, all_chunks, task_type="RETRIEVAL_DOCUMENT")
# # # #     return all_chunks, all_sources, chunk_vectors


# # # # def embed_texts(client, texts, task_type):
# # # #     result = client.models.embed_content(
# # # #         model=EMBED_MODEL,
# # # #         contents=texts,
# # # #         config=types.EmbedContentConfig(task_type=task_type),
# # # #     )
# # # #     return [np.array(e.values) for e in result.embeddings]


# # # # def cosine_similarity(a, b):
# # # #     return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# # # # def get_top_chunks(query_vector, chunk_vectors, chunks, sources, top_k=TOP_K):
# # # #     scores = [cosine_similarity(query_vector, vec) for vec in chunk_vectors]
# # # #     ranked = sorted(zip(scores, chunks, sources), key=lambda item: item[0], reverse=True)
# # # #     return [(chunk, source) for score, chunk, source in ranked[:top_k]]


# # # # def build_prompt(question, context_pairs):
# # # #     context_text = "\n\n---\n\n".join(
# # # #         f"From {source}:\n{chunk}" for chunk, source in context_pairs
# # # #     )
# # # #     prompt = (
# # # #         "You are answering a question using only the context below, taken from PDF "
# # # #         "files the user uploaded. If the answer is not in the context, say you cannot "
# # # #         "find it in the documents rather than guessing.\n\n"
# # # #         f"Context:\n{context_text}\n\n"
# # # #         f"Question: {question}\n\n"
# # # #         "Answer clearly and briefly."
# # # #     )
# # # #     return prompt


# # # # def init_session_state():
# # # #     defaults = {
# # # #         "chunks": None,
# # # #         "sources": None,
# # # #         "chunk_vectors": None,
# # # #         "indexed_file_names": None,
# # # #         "history": [],
# # # #     }
# # # #     for key, value in defaults.items():
# # # #         if key not in st.session_state:
# # # #             st.session_state[key] = value


# # # # def end_session():
# # # #     st.session_state.chunks = None
# # # #     st.session_state.sources = None
# # # #     st.session_state.chunk_vectors = None
# # # #     st.session_state.indexed_file_names = None
# # # #     st.session_state.history = []


# # # # def main():
# # # #     st.set_page_config(page_title="PDF Question Answering", page_icon="📄")

# # # #     with st.sidebar:
# # # #         st.markdown("## 📄 PDFTalker")
# # # #         st.caption("SID AI LAB")

# # # #     st.title("Ask questions about your PDFs")
# # # #     st.write("Upload one or more small PDFs, then type a question about them below.")

# # # #     init_session_state()

# # # #     api_key = st.secrets.get("GEMINI_API_KEY")
# # # #     uploaded_files = st.file_uploader(
# # # #         "Upload PDFs", type=["pdf"], accept_multiple_files=True
# # # #     )

# # # #     if not api_key:
# # # #         st.error("No Gemini API key found. Add GEMINI_API_KEY in your app's Secrets.")
# # # #         return

# # # #     client = genai.Client(api_key=api_key)

# # # #     current_names = sorted(f.name for f in uploaded_files) if uploaded_files else None
# # # #     if uploaded_files and current_names != st.session_state.indexed_file_names:
# # # #         with st.spinner("Reading and indexing your PDFs..."):
# # # #             chunks, sources, chunk_vectors = build_index_from_files(client, uploaded_files)
# # # #             if not chunks:
# # # #                 st.error("No readable text was found in these PDFs. They may be scanned images.")
# # # #                 return
# # # #             st.session_state.chunks = chunks
# # # #             st.session_state.sources = sources
# # # #             st.session_state.chunk_vectors = chunk_vectors
# # # #             st.session_state.indexed_file_names = current_names
# # # #         file_count = len(uploaded_files)
# # # #         st.success(f"Indexed {len(chunks)} chunks from {file_count} file(s).")

# # # #     if st.session_state.chunks is None:
# # # #         st.info("Upload at least one PDF to continue.")
# # # #         return

# # # #     question = st.text_input("Your question", key="question_box")

# # # #     if question:
# # # #         with st.spinner("Thinking..."):
# # # #             query_vector = embed_texts(client, [question], task_type="RETRIEVAL_QUERY")[0]
# # # #             top_pairs = get_top_chunks(
# # # #                 query_vector,
# # # #                 st.session_state.chunk_vectors,
# # # #                 st.session_state.chunks,
# # # #                 st.session_state.sources,
# # # #             )
# # # #             prompt = build_prompt(question, top_pairs)
# # # #             response = client.models.generate_content(
# # # #                 model=CHAT_MODEL,
# # # #                 contents=prompt,
# # # #             )
# # # #         answer = response.text
# # # #         st.session_state.history.append((question, answer, top_pairs))

# # # #     if st.session_state.history:
# # # #         last_question, last_answer, last_pairs = st.session_state.history[-1]
# # # #         st.markdown("### Answer")
# # # #         st.write(last_answer)

# # # #         with st.expander("Show the passages used to answer this"):
# # # #             for i, (chunk, source) in enumerate(last_pairs, start=1):
# # # #                 st.markdown(f"**Passage {i}, from {source}**")
# # # #                 st.write(chunk)

# # # #         st.markdown("### Conversation history")
# # # #         for past_question, past_answer, _ in reversed(st.session_state.history):
# # # #             st.markdown(f"**Q: {past_question}**")
# # # #             st.write(past_answer)
# # # #             st.markdown("---")

# # # #         if st.button("End session"):
# # # #             end_session()
# # # #             st.rerun()


# # # # if __name__ == "__main__":
# # # #     main()




# # # # """
# # # # Simple RAG app using Gemini.
# # # # Upload one or more small PDFs, ask a question in the input box, get an answer grounded in them.

# # # # How this works, in plain terms.
# # # # The text from every uploaded PDF is pulled out and cut into small chunks.
# # # # Each chunk is converted into a vector using the Gemini embedding model, and remembers which file it came from.
# # # # When a question comes in, it is converted into a vector too.
# # # # The chunks whose vectors are closest to the question's vector are picked as context, regardless of which file they came from.
# # # # Those chunks, along with the question, are sent to a Gemini text model to produce the final answer.
# # # # Every question and answer is kept in a running history for the session, until you choose to end it.
# # # # """

# # # # import streamlit as st
# # # # from pypdf import PdfReader
# # # # from google import genai
# # # # from google.genai import types
# # # # import numpy as np


# # # # EMBED_MODEL = "gemini-embedding-001"
# # # # CHAT_MODEL = "gemini-3.8-flash"
# # # # CHUNK_SIZE = 800
# # # # CHUNK_OVERLAP = 150
# # # # TOP_K = 4


# # # # def extract_text_from_pdf(file):
# # # #     reader = PdfReader(file)
# # # #     pages = []
# # # #     for page in reader.pages:
# # # #         text = page.extract_text() or ""
# # # #         pages.append(text)
# # # #     return "\n".join(pages)


# # # # def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
# # # #     chunks = []
# # # #     start = 0
# # # #     length = len(text)
# # # #     while start < length:
# # # #         end = min(start + chunk_size, length)
# # # #         chunk = text[start:end].strip()
# # # #         if chunk:
# # # #             chunks.append(chunk)
# # # #         start += chunk_size - overlap
# # # #     return chunks


# # # # def build_index_from_files(client, uploaded_files):
# # # #     """Read every uploaded PDF, chunk it, and embed each chunk.
# # # #     Returns a list of chunks and a matching list of source file names."""
# # # #     all_chunks = []
# # # #     all_sources = []
# # # #     for uploaded_file in uploaded_files:
# # # #         raw_text = extract_text_from_pdf(uploaded_file)
# # # #         if not raw_text.strip():
# # # #             continue
# # # #         chunks = split_into_chunks(raw_text)
# # # #         all_chunks.extend(chunks)
# # # #         all_sources.extend([uploaded_file.name] * len(chunks))
# # # #     if not all_chunks:
# # # #         return [], [], []
# # # #     chunk_vectors = embed_texts(client, all_chunks, task_type="RETRIEVAL_DOCUMENT")
# # # #     return all_chunks, all_sources, chunk_vectors


# # # # def embed_texts(client, texts, task_type):
# # # #     result = client.models.embed_content(
# # # #         model=EMBED_MODEL,
# # # #         contents=texts,
# # # #         config=types.EmbedContentConfig(task_type=task_type),
# # # #     )
# # # #     return [np.array(e.values) for e in result.embeddings]


# # # # def cosine_similarity(a, b):
# # # #     return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# # # # def get_top_chunks(query_vector, chunk_vectors, chunks, sources, top_k=TOP_K):
# # # #     scores = [cosine_similarity(query_vector, vec) for vec in chunk_vectors]
# # # #     ranked = sorted(zip(scores, chunks, sources), key=lambda item: item[0], reverse=True)
# # # #     return [(chunk, source) for score, chunk, source in ranked[:top_k]]


# # # # def build_prompt(question, context_pairs):
# # # #     context_text = "\n\n---\n\n".join(
# # # #         f"From {source}:\n{chunk}" for chunk, source in context_pairs
# # # #     )
# # # #     prompt = (
# # # #         "You are answering a question using only the context below, taken from PDF "
# # # #         "files the user uploaded. If the answer is not in the context, say you cannot "
# # # #         "find it in the documents rather than guessing.\n\n"
# # # #         f"Context:\n{context_text}\n\n"
# # # #         f"Question: {question}\n\n"
# # # #         "Answer clearly and briefly."
# # # #     )
# # # #     return prompt


# # # # def init_session_state():
# # # #     defaults = {
# # # #         "chunks": None,
# # # #         "sources": None,
# # # #         "chunk_vectors": None,
# # # #         "indexed_file_names": None,
# # # #         "history": [],
# # # #     }
# # # #     for key, value in defaults.items():
# # # #         if key not in st.session_state:
# # # #             st.session_state[key] = value


# # # # def end_session():
# # # #     st.session_state.chunks = None
# # # #     st.session_state.sources = None
# # # #     st.session_state.chunk_vectors = None
# # # #     st.session_state.indexed_file_names = None
# # # #     st.session_state.history = []


# # # # def main():
# # # #     st.set_page_config(page_title="PDF Question Answering", page_icon="📄")

# # # #     with st.sidebar:
# # # #         st.markdown(
# # # #             """
# # # #             <div style="
# # # #                 background: linear-gradient(135deg, #4F46E5, #7C3AED);
# # # #                 padding: 18px 16px;
# # # #                 border-radius: 12px;
# # # #                 margin-bottom: 12px;
# # # #             ">
# # # #                 <div style="font-size: 28px; line-height: 1;">📄</div>
# # # #                 <div style="color: white; font-size: 20px; font-weight: 700; margin-top: 6px;">
# # # #                     PDFTalker
# # # #                 </div>
# # # #                 <div style="color: rgba(255,255,255,0.75); font-size: 12px; letter-spacing: 1px; margin-top: 2px;">
# # # #                     SID AI LAB
# # # #                 </div>
# # # #             </div>
# # # #             """,
# # # #             unsafe_allow_html=True,
# # # #         )

# # # #     st.title("Ask questions about your PDFs")
# # # #     st.write("Upload one or more small PDFs, then type a question about them below.")

# # # #     init_session_state()

# # # #     api_key = st.secrets.get("GEMINI_API_KEY")
# # # #     uploaded_files = st.file_uploader(
# # # #         "Upload PDFs", type=["pdf"], accept_multiple_files=True
# # # #     )

# # # #     if not api_key:
# # # #         st.error("No Gemini API key found. Add GEMINI_API_KEY in your app's Secrets.")
# # # #         return

# # # #     client = genai.Client(api_key=api_key)

# # # #     current_names = sorted(f.name for f in uploaded_files) if uploaded_files else None
# # # #     if uploaded_files and current_names != st.session_state.indexed_file_names:
# # # #         with st.spinner("Reading and indexing your PDFs..."):
# # # #             chunks, sources, chunk_vectors = build_index_from_files(client, uploaded_files)
# # # #             if not chunks:
# # # #                 st.error("No readable text was found in these PDFs. They may be scanned images.")
# # # #                 return
# # # #             st.session_state.chunks = chunks
# # # #             st.session_state.sources = sources
# # # #             st.session_state.chunk_vectors = chunk_vectors
# # # #             st.session_state.indexed_file_names = current_names
# # # #         file_count = len(uploaded_files)
# # # #         st.success(f"Indexed {len(chunks)} chunks from {file_count} file(s).")

# # # #     if st.session_state.chunks is None:
# # # #         st.info("Upload at least one PDF to continue.")
# # # #         return

# # # #     question = st.text_input("Your question", key="question_box")

# # # #     if question:
# # # #         with st.spinner("Thinking..."):
# # # #             query_vector = embed_texts(client, [question], task_type="RETRIEVAL_QUERY")[0]
# # # #             top_pairs = get_top_chunks(
# # # #                 query_vector,
# # # #                 st.session_state.chunk_vectors,
# # # #                 st.session_state.chunks,
# # # #                 st.session_state.sources,
# # # #             )
# # # #             prompt = build_prompt(question, top_pairs)
# # # #             response = client.models.generate_content(
# # # #                 model=CHAT_MODEL,
# # # #                 contents=prompt,
# # # #             )
# # # #         answer = response.text
# # # #         st.session_state.history.append((question, answer, top_pairs))

# # # #     if st.session_state.history:
# # # #         last_question, last_answer, last_pairs = st.session_state.history[-1]
# # # #         st.markdown("### Answer")
# # # #         st.write(last_answer)

# # # #         with st.expander("Show the passages used to answer this"):
# # # #             for i, (chunk, source) in enumerate(last_pairs, start=1):
# # # #                 st.markdown(f"**Passage {i}, from {source}**")
# # # #                 st.write(chunk)

# # # #         st.markdown("### Conversation history")
# # # #         for past_question, past_answer, _ in reversed(st.session_state.history):
# # # #             st.markdown(f"**Q: {past_question}**")
# # # #             st.write(past_answer)
# # # #             st.markdown("---")

# # # #         if st.button("End session"):
# # # #             end_session()
# # # #             st.rerun()


# # # # if __name__ == "__main__":
# # # #     main()


# # # """
# # # Simple RAG app using Gemini.
# # # Upload one or more small PDFs, ask a question in the input box, get an answer grounded in them.

# # # How this works, in plain terms.
# # # The text from every uploaded PDF is pulled out and cut into small chunks.
# # # Each chunk is converted into a vector using the Gemini embedding model, and remembers which file it came from.
# # # When a question comes in, it is converted into a vector too.
# # # The chunks whose vectors are closest to the question's vector are picked as context, regardless of which file they came from.
# # # Those chunks, along with the question, are sent to a Gemini text model to produce the final answer.
# # # Every question and answer is kept in a running history for the session, until you choose to end it.
# # # """

# # # import streamlit as st
# # # from pypdf import PdfReader
# # # from google import genai
# # # from google.genai import types
# # # import numpy as np


# # # EMBED_MODEL = "gemini-embedding-001"
# # # CHAT_MODEL = "gemini-3.8-flash"
# # # CHUNK_SIZE = 800
# # # CHUNK_OVERLAP = 150
# # # TOP_K = 4


# # # def extract_text_from_pdf(file):
# # #     reader = PdfReader(file)
# # #     pages = []
# # #     for page in reader.pages:
# # #         text = page.extract_text() or ""
# # #         pages.append(text)
# # #     return "\n".join(pages)


# # # def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
# # #     chunks = []
# # #     start = 0
# # #     length = len(text)
# # #     while start < length:
# # #         end = min(start + chunk_size, length)
# # #         chunk = text[start:end].strip()
# # #         if chunk:
# # #             chunks.append(chunk)
# # #         start += chunk_size - overlap
# # #     return chunks


# # # def build_index_from_files(client, uploaded_files):
# # #     """Read every uploaded PDF, chunk it, and embed each chunk.
# # #     Returns a list of chunks and a matching list of source file names."""
# # #     all_chunks = []
# # #     all_sources = []
# # #     for uploaded_file in uploaded_files:
# # #         raw_text = extract_text_from_pdf(uploaded_file)
# # #         if not raw_text.strip():
# # #             continue
# # #         chunks = split_into_chunks(raw_text)
# # #         all_chunks.extend(chunks)
# # #         all_sources.extend([uploaded_file.name] * len(chunks))
# # #     if not all_chunks:
# # #         return [], [], []
# # #     chunk_vectors = embed_texts(client, all_chunks, task_type="RETRIEVAL_DOCUMENT")
# # #     return all_chunks, all_sources, chunk_vectors


# # # def embed_texts(client, texts, task_type):
# # #     result = client.models.embed_content(
# # #         model=EMBED_MODEL,
# # #         contents=texts,
# # #         config=types.EmbedContentConfig(task_type=task_type),
# # #     )
# # #     return [np.array(e.values) for e in result.embeddings]


# # # def cosine_similarity(a, b):
# # #     return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# # # def get_top_chunks(query_vector, chunk_vectors, chunks, sources, top_k=TOP_K):
# # #     scores = [cosine_similarity(query_vector, vec) for vec in chunk_vectors]
# # #     ranked = sorted(zip(scores, chunks, sources), key=lambda item: item[0], reverse=True)
# # #     return [(chunk, source) for score, chunk, source in ranked[:top_k]]


# # # def build_prompt(question, context_pairs):
# # #     context_text = "\n\n---\n\n".join(
# # #         f"From {source}:\n{chunk}" for chunk, source in context_pairs
# # #     )
# # #     prompt = (
# # #         "You are answering a question using only the context below, taken from PDF "
# # #         "files the user uploaded. If the answer is not in the context, say you cannot "
# # #         "find it in the documents rather than guessing.\n\n"
# # #         f"Context:\n{context_text}\n\n"
# # #         f"Question: {question}\n\n"
# # #         "Answer clearly and briefly."
# # #     )
# # #     return prompt


# # # def init_session_state():
# # #     defaults = {
# # #         "chunks": None,
# # #         "sources": None,
# # #         "chunk_vectors": None,
# # #         "indexed_file_names": None,
# # #         "history": [],
# # #     }
# # #     for key, value in defaults.items():
# # #         if key not in st.session_state:
# # #             st.session_state[key] = value


# # # def end_session():
# # #     st.session_state.chunks = None
# # #     st.session_state.sources = None
# # #     st.session_state.chunk_vectors = None
# # #     st.session_state.indexed_file_names = None
# # #     st.session_state.history = []


# # # def main():
# # #     st.set_page_config(page_title="PDF Question Answering", page_icon="📄")

# # #     with st.sidebar:
# # #         st.image("pdftalker_logo.png", use_container_width=True)

# # #     st.title("Ask questions about your PDFs")
# # #     st.write("Upload one or more small PDFs, then type a question about them below.")

# # #     init_session_state()

# # #     api_key = st.secrets.get("GEMINI_API_KEY")
# # #     uploaded_files = st.file_uploader(
# # #         "Upload PDFs", type=["pdf"], accept_multiple_files=True
# # #     )

# # #     if not api_key:
# # #         st.error("No Gemini API key found. Add GEMINI_API_KEY in your app's Secrets.")
# # #         return

# # #     client = genai.Client(api_key=api_key)

# # #     current_names = sorted(f.name for f in uploaded_files) if uploaded_files else None
# # #     if uploaded_files and current_names != st.session_state.indexed_file_names:
# # #         with st.spinner("Reading and indexing your PDFs..."):
# # #             chunks, sources, chunk_vectors = build_index_from_files(client, uploaded_files)
# # #             if not chunks:
# # #                 st.error("No readable text was found in these PDFs. They may be scanned images.")
# # #                 return
# # #             st.session_state.chunks = chunks
# # #             st.session_state.sources = sources
# # #             st.session_state.chunk_vectors = chunk_vectors
# # #             st.session_state.indexed_file_names = current_names
# # #         file_count = len(uploaded_files)
# # #         st.success(f"Indexed {len(chunks)} chunks from {file_count} file(s).")

# # #     if st.session_state.chunks is None:
# # #         st.info("Upload at least one PDF to continue.")
# # #         return

# # #     question = st.text_input("Your question", key="question_box")

# # #     if question:
# # #         with st.spinner("Thinking..."):
# # #             query_vector = embed_texts(client, [question], task_type="RETRIEVAL_QUERY")[0]
# # #             top_pairs = get_top_chunks(
# # #                 query_vector,
# # #                 st.session_state.chunk_vectors,
# # #                 st.session_state.chunks,
# # #                 st.session_state.sources,
# # #             )
# # #             prompt = build_prompt(question, top_pairs)
# # #             response = client.models.generate_content(
# # #                 model=CHAT_MODEL,
# # #                 contents=prompt,
# # #             )
# # #         answer = response.text
# # #         st.session_state.history.append((question, answer, top_pairs))

# # #     if st.session_state.history:
# # #         last_question, last_answer, last_pairs = st.session_state.history[-1]
# # #         st.markdown("### Answer")
# # #         st.write(last_answer)

# # #         with st.expander("Show the passages used to answer this"):
# # #             for i, (chunk, source) in enumerate(last_pairs, start=1):
# # #                 st.markdown(f"**Passage {i}, from {source}**")
# # #                 st.write(chunk)

# # #         st.markdown("### Conversation history")
# # #         for past_question, past_answer, _ in reversed(st.session_state.history):
# # #             st.markdown(f"**Q: {past_question}**")
# # #             st.write(past_answer)
# # #             st.markdown("---")

# # #         if st.button("End session"):
# # #             end_session()
# # #             st.rerun()


# # # if __name__ == "__main__":
# # #     main()



# # """
# # Simple RAG app using Gemini.
# # Upload one or more small PDFs, ask a question in the input box, get an answer grounded in them.

# # How this works, in plain terms.
# # The text from every uploaded PDF is pulled out and cut into small chunks.
# # Each chunk is converted into a vector using the Gemini embedding model, and remembers which file it came from.
# # When a question comes in, it is converted into a vector too.
# # The chunks whose vectors are closest to the question's vector are picked as context, regardless of which file they came from.
# # Those chunks, along with the question, are sent to a Gemini text model to produce the final answer.
# # Every question and answer is kept in a running history for the session, until you choose to end it.
# # """

# # import streamlit as st
# # from pypdf import PdfReader
# # import fitz  # PyMuPDF
# # from google import genai
# # from google.genai import types
# # import numpy as np


# # EMBED_MODEL = "gemini-embedding-001"
# # CHAT_MODEL = "gemini-3.5-flash"
# # CHUNK_SIZE = 800
# # CHUNK_OVERLAP = 150
# # TOP_K = 4


# # def extract_text_with_pymupdf(file):
# #     file.seek(0)
# #     doc = fitz.open(stream=file.read(), filetype="pdf")
# #     pages = [page.get_text() for page in doc]
# #     doc.close()
# #     return "\n".join(pages)


# # def extract_text_from_pdf(file):
# #     """Try pypdf first. Some PDFs have unusual font data that makes pypdf fail,
# #     so if that happens, fall back to PyMuPDF which handles those files."""
# #     try:
# #         file.seek(0)
# #         reader = PdfReader(file)
# #         pages = []
# #         for page in reader.pages:
# #             text = page.extract_text() or ""
# #             pages.append(text)
# #         return "\n".join(pages)
# #     except Exception:
# #         return extract_text_with_pymupdf(file)


# # def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
# #     chunks = []
# #     start = 0
# #     length = len(text)
# #     while start < length:
# #         end = min(start + chunk_size, length)
# #         chunk = text[start:end].strip()
# #         if chunk:
# #             chunks.append(chunk)
# #         start += chunk_size - overlap
# #     return chunks


# # def build_index_from_files(client, uploaded_files):
# #     """Read every uploaded PDF, chunk it, and embed each chunk.
# #     Returns a list of chunks and a matching list of source file names."""
# #     all_chunks = []
# #     all_sources = []
# #     for uploaded_file in uploaded_files:
# #         raw_text = extract_text_from_pdf(uploaded_file)
# #         if not raw_text.strip():
# #             continue
# #         chunks = split_into_chunks(raw_text)
# #         all_chunks.extend(chunks)
# #         all_sources.extend([uploaded_file.name] * len(chunks))
# #     if not all_chunks:
# #         return [], [], []
# #     chunk_vectors = embed_texts(client, all_chunks, task_type="RETRIEVAL_DOCUMENT")
# #     return all_chunks, all_sources, chunk_vectors


# # def embed_texts(client, texts, task_type):
# #     result = client.models.embed_content(
# #         model=EMBED_MODEL,
# #         contents=texts,
# #         config=types.EmbedContentConfig(task_type=task_type),
# #     )
# #     return [np.array(e.values) for e in result.embeddings]


# # def cosine_similarity(a, b):
# #     return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# # def get_top_chunks(query_vector, chunk_vectors, chunks, sources, top_k=TOP_K):
# #     scores = [cosine_similarity(query_vector, vec) for vec in chunk_vectors]
# #     ranked = sorted(zip(scores, chunks, sources), key=lambda item: item[0], reverse=True)
# #     return [(chunk, source) for score, chunk, source in ranked[:top_k]]


# # def build_prompt(question, context_pairs):
# #     context_text = "\n\n---\n\n".join(
# #         f"From {source}:\n{chunk}" for chunk, source in context_pairs
# #     )
# #     prompt = (
# #         "You are answering a question using only the context below, taken from PDF "
# #         "files the user uploaded. If the answer is not in the context, say you cannot "
# #         "find it in the documents rather than guessing.\n\n"
# #         f"Context:\n{context_text}\n\n"
# #         f"Question: {question}\n\n"
# #         "Answer clearly and briefly."
# #     )
# #     return prompt


# # def init_session_state():
# #     defaults = {
# #         "chunks": None,
# #         "sources": None,
# #         "chunk_vectors": None,
# #         "indexed_file_names": None,
# #         "history": [],
# #     }
# #     for key, value in defaults.items():
# #         if key not in st.session_state:
# #             st.session_state[key] = value


# # def end_session():
# #     st.session_state.chunks = None
# #     st.session_state.sources = None
# #     st.session_state.chunk_vectors = None
# #     st.session_state.indexed_file_names = None
# #     st.session_state.history = []


# # def main():
# #     st.set_page_config(page_title="PDF Question Answering", page_icon="📄")

# #     with st.sidebar:
# #         st.image("pdftalker_logo.png", width="stretch")

# #     st.title("Ask questions about your PDFs")
# #     st.write("Upload one or more small PDFs, then type a question about them below.")

# #     init_session_state()

# #     api_key = st.secrets.get("GEMINI_API_KEY")
# #     uploaded_files = st.file_uploader(
# #         "Upload PDFs", type=["pdf"], accept_multiple_files=True
# #     )

# #     if not api_key:
# #         st.error("No Gemini API key found. Add GEMINI_API_KEY in your app's Secrets.")
# #         return

# #     client = genai.Client(api_key=api_key)

# #     current_names = sorted(f.name for f in uploaded_files) if uploaded_files else None
# #     if uploaded_files and current_names != st.session_state.indexed_file_names:
# #         with st.spinner("Reading and indexing your PDFs..."):
# #             chunks, sources, chunk_vectors = build_index_from_files(client, uploaded_files)
# #             if not chunks:
# #                 st.error("No readable text was found in these PDFs. They may be scanned images.")
# #                 return
# #             st.session_state.chunks = chunks
# #             st.session_state.sources = sources
# #             st.session_state.chunk_vectors = chunk_vectors
# #             st.session_state.indexed_file_names = current_names
# #         file_count = len(uploaded_files)
# #         st.success(f"Indexed {len(chunks)} chunks from {file_count} file(s).")

# #     if st.session_state.chunks is None:
# #         st.info("Upload at least one PDF to continue.")
# #         return

# #     question = st.text_input("Your question", key="question_box")

# #     if question:
# #         with st.spinner("Thinking..."):
# #             query_vector = embed_texts(client, [question], task_type="RETRIEVAL_QUERY")[0]
# #             top_pairs = get_top_chunks(
# #                 query_vector,
# #                 st.session_state.chunk_vectors,
# #                 st.session_state.chunks,
# #                 st.session_state.sources,
# #             )
# #             prompt = build_prompt(question, top_pairs)
# #             response = client.models.generate_content(
# #                 model=CHAT_MODEL,
# #                 contents=prompt,
# #             )
# #         answer = response.text
# #         st.session_state.history.append((question, answer, top_pairs))

# #     if st.session_state.history:
# #         last_question, last_answer, last_pairs = st.session_state.history[-1]
# #         st.markdown("### Answer")
# #         st.write(last_answer)

# #         with st.expander("Show the passages used to answer this"):
# #             for i, (chunk, source) in enumerate(last_pairs, start=1):
# #                 st.markdown(f"**Passage {i}, from {source}**")
# #                 st.write(chunk)

# #         st.markdown("### Conversation history")
# #         for past_question, past_answer, _ in reversed(st.session_state.history):
# #             st.markdown(f"**Q: {past_question}**")
# #             st.write(past_answer)
# #             st.markdown("---")

# #         if st.button("End session"):
# #             end_session()
# #             st.rerun()


# # if __name__ == "__main__":
# #     main()



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
# import fitz  # PyMuPDF
# from google import genai
# from google.genai import types
# import numpy as np
# import pandas as pd
# import altair as alt


# EMBED_MODEL = "gemini-embedding-001"
# CHAT_MODEL = "gemini-3.5-flash"
# CHUNK_SIZE = 800
# CHUNK_OVERLAP = 150
# TOP_K = 4


# def extract_text_with_pymupdf(file):
#     file.seek(0)
#     doc = fitz.open(stream=file.read(), filetype="pdf")
#     pages = [page.get_text() for page in doc]
#     doc.close()
#     return "\n".join(pages)


# def extract_text_from_pdf(file):
#     """Try pypdf first. Some PDFs have unusual font data that makes pypdf fail,
#     so if that happens, fall back to PyMuPDF which handles those files."""
#     try:
#         file.seek(0)
#         reader = PdfReader(file)
#         pages = []
#         for page in reader.pages:
#             text = page.extract_text() or ""
#             pages.append(text)
#         return "\n".join(pages)
#     except Exception:
#         return extract_text_with_pymupdf(file)


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


# # ---------------------------------------------------------------------------
# # Extra functions that only power the "see how RAG works" displays.
# # They read data the app already has and never change how retrieval or answers work.
# # ---------------------------------------------------------------------------

# def project_to_2d(chunk_vectors, query_vector):
#     """Flatten long vectors down to 2 numbers each so they can be drawn on a chart.
#     This uses PCA, which keeps the directions along which the chunks differ the most."""
#     data = np.vstack(chunk_vectors)
#     mean = data.mean(axis=0)
#     centered = data - mean
#     _, _, vt = np.linalg.svd(centered, full_matrices=False)
#     components = vt[:2]
#     chunk_points = centered @ components.T
#     query_point = (query_vector - mean) @ components.T
#     return chunk_points, query_point


# def build_details(question, query_vector, prompt, chunk_vectors, chunks, sources, top_k=TOP_K):
#     """Collect everything the demo displays need: match scores, the exact prompt,
#     and the points for the document map."""
#     scores = [cosine_similarity(query_vector, vec) for vec in chunk_vectors]
#     top_idx = [int(i) for i in np.argsort(scores)[::-1][:top_k]]
#     top_scores = [scores[i] for i in top_idx]

#     map_df = None
#     if len(chunk_vectors) >= 3:
#         chunk_points, query_point = project_to_2d(chunk_vectors, query_vector)
#         top_set = set(top_idx)
#         rows = []
#         for i, (x, y) in enumerate(chunk_points):
#             rows.append(
#                 {
#                     "x": float(x),
#                     "y": float(y),
#                     "group": "Retrieved passage" if i in top_set else "Other chunk",
#                     "source": sources[i],
#                     "score": f"{scores[i]:.3f}",
#                     "preview": chunks[i][:140].replace("\n", " "),
#                 }
#             )
#         rows.append(
#             {
#                 "x": float(query_point[0]),
#                 "y": float(query_point[1]),
#                 "group": "Your question",
#                 "source": "",
#                 "score": "",
#                 "preview": question[:140],
#             }
#         )
#         map_df = pd.DataFrame(rows)

#     return {
#         "top_scores": top_scores,
#         "prompt": prompt,
#         "map_df": map_df,
#         "dimensions": len(query_vector),
#     }


# def render_document_map(map_df, dimensions):
#     domain = ["Other chunk", "Retrieved passage", "Your question"]
#     chart = (
#         alt.Chart(map_df)
#         .mark_point(filled=True, opacity=0.85)
#         .encode(
#             x=alt.X("x:Q", axis=None),
#             y=alt.Y("y:Q", axis=None),
#             color=alt.Color(
#                 "group:N",
#                 scale=alt.Scale(domain=domain, range=["#94A3B8", "#F59E0B", "#EF4444"]),
#                 legend=alt.Legend(title=None, orient="bottom"),
#             ),
#             size=alt.Size(
#                 "group:N",
#                 scale=alt.Scale(domain=domain, range=[70, 260, 420]),
#                 legend=None,
#             ),
#             shape=alt.Shape(
#                 "group:N",
#                 scale=alt.Scale(domain=domain, range=["circle", "circle", "diamond"]),
#                 legend=None,
#             ),
#             tooltip=[
#                 alt.Tooltip("group:N", title="Type"),
#                 alt.Tooltip("source:N", title="File"),
#                 alt.Tooltip("score:N", title="Match score"),
#                 alt.Tooltip("preview:N", title="Text"),
#             ],
#         )
#         .properties(height=420)
#         .interactive()
#     )
#     st.altair_chart(chart, width="stretch")
#     st.caption(
#         f"Every chunk of your documents is a list of {dimensions} numbers that captures its meaning. "
#         "This map flattens those lists down to two numbers so they can be drawn. "
#         "Chunks that sit close together say similar things. The large diamond is your question, "
#         "and the amber dots are the passages that landed closest to it, which is exactly why "
#         "they were picked."
#     )


# def init_session_state():
#     defaults = {
#         "chunks": None,
#         "sources": None,
#         "chunk_vectors": None,
#         "indexed_file_names": None,
#         "history": [],
#         "last_details": None,
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
#     st.session_state.last_details = None


# def main():
#     st.set_page_config(page_title="PDF Question Answering", page_icon="📄")

#     with st.sidebar:
#         st.image("pdftalker_logo.png", width="stretch")

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
#         try:
#             st.session_state.last_details = build_details(
#                 question,
#                 query_vector,
#                 prompt,
#                 st.session_state.chunk_vectors,
#                 st.session_state.chunks,
#                 st.session_state.sources,
#             )
#         except Exception:
#             st.session_state.last_details = None
#         st.session_state.history.append((question, answer, top_pairs))

#     if st.session_state.history:
#         last_question, last_answer, last_pairs = st.session_state.history[-1]
#         st.markdown("### Answer")
#         st.write(last_answer)

#         details = st.session_state.last_details

#         with st.expander("Show the passages used to answer this"):
#             for i, (chunk, source) in enumerate(last_pairs, start=1):
#                 st.markdown(f"**Passage {i}, from {source}**")
#                 if details and i <= len(details["top_scores"]):
#                     score = details["top_scores"][i - 1]
#                     st.progress(min(max(score, 0.0), 1.0), text=f"Match score: {score:.3f}")
#                 st.write(chunk)

#         if details:
#             with st.expander("Show the exact prompt sent to Gemini"):
#                 st.caption(
#                     "This is the full text the model receives. The passages above are placed "
#                     "inside it next to your question, which is the augmented part of retrieval "
#                     "augmented generation."
#                 )
#                 st.code(details["prompt"], language=None, wrap_lines=True)

#             if details["map_df"] is not None:
#                 st.markdown("### Document map")
#                 render_document_map(details["map_df"], details["dimensions"])
#             else:
#                 st.caption("The document map appears once your PDFs have at least three chunks.")

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
import fitz  # PyMuPDF
from google import genai
from google.genai import types
import numpy as np
import pandas as pd
import altair as alt


EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4


def extract_text_with_pymupdf(file):
    file.seek(0)
    doc = fitz.open(stream=file.read(), filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


def extract_text_from_pdf(file):
    """Try pypdf first. Some PDFs have unusual font data that makes pypdf fail,
    so if that happens, fall back to PyMuPDF which handles those files."""
    try:
        file.seek(0)
        reader = PdfReader(file)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n".join(pages)
    except Exception:
        return extract_text_with_pymupdf(file)


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


# ---------------------------------------------------------------------------
# Extra functions that only power the "see how RAG works" displays.
# They read data the app already has and never change how retrieval or answers work.
# ---------------------------------------------------------------------------

def project_to_2d(chunk_vectors, query_vector):
    """Flatten long vectors down to 2 numbers each so they can be drawn on a chart.
    This uses PCA, which keeps the directions along which the chunks differ the most."""
    data = np.vstack(chunk_vectors)
    mean = data.mean(axis=0)
    centered = data - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    chunk_points = centered @ components.T
    query_point = (query_vector - mean) @ components.T
    return chunk_points, query_point


def build_details(question, query_vector, prompt, chunk_vectors, chunks, sources, top_k=TOP_K):
    """Collect everything the demo displays need: match scores, the exact prompt,
    and the points for the document map."""
    scores = [cosine_similarity(query_vector, vec) for vec in chunk_vectors]
    top_idx = [int(i) for i in np.argsort(scores)[::-1][:top_k]]
    top_scores = [scores[i] for i in top_idx]

    map_df = None
    if len(chunk_vectors) >= 3:
        chunk_points, query_point = project_to_2d(chunk_vectors, query_vector)
        top_set = set(top_idx)
        rows = []
        for i, (x, y) in enumerate(chunk_points):
            rows.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "group": "Retrieved passage" if i in top_set else "Other chunk",
                    "source": sources[i],
                    "score": f"{scores[i]:.3f}",
                    "preview": chunks[i][:140].replace("\n", " "),
                }
            )
        rows.append(
            {
                "x": float(query_point[0]),
                "y": float(query_point[1]),
                "group": "Your question",
                "source": "",
                "score": "",
                "preview": question[:140],
            }
        )
        map_df = pd.DataFrame(rows)

    return {
        "top_scores": top_scores,
        "prompt": prompt,
        "map_df": map_df,
        "dimensions": len(query_vector),
    }


def render_document_map(map_df, dimensions):
    domain = ["Other chunk", "Retrieved passage", "Your question"]
    chart = (
        alt.Chart(map_df)
        .mark_point(filled=True, opacity=0.85)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(domain=domain, range=["#94A3B8", "#F59E0B", "#EF4444"]),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            size=alt.Size(
                "group:N",
                scale=alt.Scale(domain=domain, range=[70, 260, 420]),
                legend=None,
            ),
            shape=alt.Shape(
                "group:N",
                scale=alt.Scale(domain=domain, range=["circle", "circle", "diamond"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("group:N", title="Type"),
                alt.Tooltip("source:N", title="File"),
                alt.Tooltip("score:N", title="Match score"),
                alt.Tooltip("preview:N", title="Text"),
            ],
        )
        .properties(height=420)
        .interactive()
    )
    st.altair_chart(chart, width="stretch")
    st.caption(
        f"Every chunk of your documents is a list of {dimensions} numbers that captures its meaning. "
        "This map flattens those lists down to two numbers so they can be drawn. "
        "Chunks that sit close together say similar things. The large diamond is your question, "
        "and the amber dots are the passages that landed closest to it, which is exactly why "
        "they were picked."
    )


def init_session_state():
    defaults = {
        "chunks": None,
        "sources": None,
        "chunk_vectors": None,
        "indexed_file_names": None,
        "history": [],
        "last_details": None,
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
    st.session_state.last_details = None


def main():
    st.set_page_config(page_title="PDF Question Answering", page_icon="📄")

    with st.sidebar:
        st.image("pdftalker_logo.png", width="stretch")

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
        try:
            st.session_state.last_details = build_details(
                question,
                query_vector,
                prompt,
                st.session_state.chunk_vectors,
                st.session_state.chunks,
                st.session_state.sources,
            )
        except Exception:
            st.session_state.last_details = None
        st.session_state.history.append((question, answer, top_pairs))

    if st.session_state.history:
        last_question, last_answer, last_pairs = st.session_state.history[-1]
        st.markdown("### Answer")
        st.write(last_answer)

        details = st.session_state.last_details

        with st.expander("Show the passages used to answer this"):
            for i, (chunk, source) in enumerate(last_pairs, start=1):
                st.markdown(f"**Passage {i}, from {source}**")
                if details and i <= len(details["top_scores"]):
                    score = details["top_scores"][i - 1]
                    st.progress(min(max(score, 0.0), 1.0), text=f"Match score: {score:.3f}")
                st.write(chunk)

        if details:
            if details["map_df"] is not None:
                st.markdown("### Document map")
                render_document_map(details["map_df"], details["dimensions"])
            else:
                st.caption("The document map appears once your PDFs have at least three chunks.")

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
