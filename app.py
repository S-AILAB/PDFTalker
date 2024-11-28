import gradio as gr
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("Google API Key not found. Please check your .env file.")
genai.configure(api_key=google_api_key)

# Initialize conversation history
conversation_history = {}

# Function Definitions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Make sure to safely load the vector store, or regenerate it if it doesn't exist
    if not os.path.exists("faiss_index"):
        return "No vector store found. Please upload and process a PDF file first."

    # Load the vector store with deserialization allowed
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    try:
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        if response and "output_text" in response:
            return response["output_text"]
        else:
            return "Sorry, no valid answer found in the context."
    except Exception as e:
        return f"An error occurred: {e}"

# Function to process PDFs and user's question
def process_and_ask_question(pdf_docs, user_question):
    if not pdf_docs:
        return "Please upload at least one PDF file.", []

    try:
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

        # Process the user's question
        answer = user_input(user_question)
        
        # Update conversation history
        conversation_history[user_question] = answer
        
        # Prepare history display
        history = "\n".join([f"Q: {q}\nA: {a}" for q, a in conversation_history.items()])
        
        return answer, history
    except Exception as e:
        return f"An error occurred: {e}", []

# Function to end session
def end_session():
    global conversation_history
    conversation_history = {}
    return "Session ended. Conversation history cleared."

# Gradio Interface
def main():
    with gr.Blocks(css=""" 
        body {background-color: #f7f7f7; font-family: Arial, sans-serif;}
        .container {max-width: 900px; margin: auto;}
        .header {font-size: 2.5em; font-weight: bold; color: #ff5733; text-align: left; animation: fadeIn 2s ease-in;}
        .subheader {font-size: 1.2em; color: #555; font-style: italic; text-align: left; animation: fadeIn 2s ease-in}
        .upload-box {border: 1px dashed #ccc; padding: 0.5em; text-align: center; font-size: 1em; color: #999; margin-bottom: 1em; display: block; margin-left: auto; margin-right: auto;  position: relative;}
        .upload-box i {position: absolute; left: 10px; top: 50%; transform: translateY(-50%); font-size: 1.5em; color: #ff5733;}
        .upload-btn {width: 100%; padding: 1em; font-size: 1em; background-color: #ff5733; color: white; border: none; cursor: pointer;}
        .footer {display: flex; justify-content: space-between; align-items: center; font-weight: bold; padding: 1em 0;}
        .footer .left {text-align: left;}
        .footer .center {flex: 1; text-align: center;}
        .footer .right {text-align: right;}
        .footer a {color: #ff5733; text-decoration: none; margin-left: 1em; transition: color 0.3s ease;}
        .footer a:hover {color: #555;}
        .history-box {width: 100%; padding: 2em; text-align: center; font-size: 1.2em; color: #999; margin-bottom: 1em;}

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
            40% { transform: translateY(-10px); }
            60% { transform: translateY(-5px); }
        }
    """) as demo:

        gr.HTML("<div class='header'>PDFTALKER</div>")
        gr.HTML("<div class='subheader'>Chat with multiple PDFs in one go</div>")

        with gr.Row():
            pdf_file_input = gr.File(label="Upload multiple PDFs together", file_count="multiple", file_types=[".pdf"], elem_classes=["upload-box"])

        with gr.Row():
            with gr.Column(elem_classes=["column-left"]):
                user_question_input = gr.Textbox(label="Ask a Question from the PDF Files", placeholder="Enter your question here...")
            with gr.Column(elem_classes=["column-right"]):
                output = gr.Textbox(label="Answer", interactive=False, placeholder="Answer is displayed here...")

        # Add the conversation history box back
        with gr.Row():
            conversation_history_box = gr.Textbox(label="Your Conversation History", interactive=False, lines=10, elem_classes=["history-box"])

        with gr.Row():
            # End session button below the conversation history
            end_session_button = gr.Button("End Session", elem_classes=["upload-btn"])

        # Add submit functionality
        user_question_input.submit(
            process_and_ask_question, inputs=[pdf_file_input, user_question_input], outputs=[output, conversation_history_box]
        )

        # Add end session functionality
        end_session_button.click(
            end_session, inputs=[], outputs=[conversation_history_box]
        )

        # Add footer
        gr.HTML("""
        <div class='footer'>
            <div class='left'>Siddharth Jain</div>
            <div class='center'>GEMINI</div>
            <div class='right'>
                <a href='https://www.linkedin.com/in/siddharth-jain-8b56a2321/' target='_blank'>🔗 LinkedIn</a>
                <a href='https://github.com/S-AILAB' target='_blank'>🐙 GitHub</a>
            </div>
        </div>
        """)

    demo.launch()

if __name__ == "__main__":
    main()
