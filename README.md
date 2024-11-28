

<h1>PDFTalker</h1>
<p><strong>PDFTalker</strong> is an innovative chatbot application designed to interact with multiple PDF documents simultaneously, providing users with a seamless experience to ask questions and get detailed answers based on the content of the uploaded PDFs.</p>

<h2>Features</h2>
<ul>
    <li>Upload and process multiple PDF files at once.</li>
    <li>Extract text from PDFs and split it into manageable chunks.</li>
    <li>Use advanced AI embeddings to create a searchable vector store.</li>
    <li>Ask questions and get detailed answers based on the PDF content.</li>
    <li>Maintain a conversation history for reference.</li>
</ul>

<h2>Installation</h2>
<p>To install and run PDFTalker locally, follow these steps:</p>
<pre>
<code>
# Clone the repository
git clone https://github.com/yourusername/PDFTalker.git

# Navigate into the project directory
cd PDFTalker

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`

# Install the required packages
pip install -r requirements.txt
</code>
</pre>

<h2>Usage</h2>
<p>To launch the PDFTalker application, simply run:</p>
<pre>
<code>
python app.py
</code>
</pre>
<p>This will start a local server. Open your web browser and navigate to <code>http LINK </code> to access the application.</p>

<h2>Configuration</h2>
<p>Create a <code>.env</code> file in the project root directory with your Google API key:</p>
<pre>
<code>
GOOGLE_API_KEY=your_google_api_key
</code>
</pre>

<h2>How It Works</h2>
<ol>
    <li>Upload one or more PDF files using the provided interface.</li>
    <li>Enter your question in the text box and press <strong>Enter</strong> to submit.</li>
    <li>The application will process the PDFs, extract text, and use AI embeddings to find the most relevant sections to answer your question.</li>
    <li>The answer will be displayed, and the conversation history will be updated below the answer box.</li>
    <li>You can click the "End Session" button to clear the conversation history and end the session.</li>
</ol>

<h2>Screenshot 1: Webpage Loaded</h2>
<p>Here is how the webpage looks when it's first loaded. You can enter your question in the text box and press <strong>Enter</strong> to submit.</p>
<img src="images\homepage.png" alt="Webpage Loaded" style="width:100%; max-width:800px;">

<h2>Screenshot 2: Model Searching for Answer</h2>
<p>This is the view when the model is processing your question and searching for the relevant information from the uploaded PDFs.</p>
<img src="images\processing the question.png" alt="Model Searching for Answer" style="width:100%; max-width:800px;">

<h2>Screenshot 3: End Session Button and Conversation Cleared</h2>
<p>Click the "End Session" button to clear the conversation history. The session will end, and all previous conversation data will be deleted.</p>
<img src="images\end session message.png" alt="End Session Cleared" style="width:100%; max-width:800px;">

<h2>Acknowledgements</h2>
<p>This project leverages several open-source libraries and APIs:</p>
<ul>
    <li><a href="https://github.com/gradio-app/gradio">Gradio</a> for the user interface.</li>
    <li><a href="https://pypi.org/project/PyPDF2/">PyPDF2</a> for PDF text extraction.</li>
    <li><a href="https://github.com/hwchase17/langchain">LangChain</a> for language model operations.</li>
    <li><a href="https://pypi.org/project/faiss-cpu/">FAISS</a> for vector search operations.</li>
    <li><a href="https://developers.generativeai.google">Google Generative AI</a> for embeddings and conversational models.</li>
    <li><a href="https://github.com/theskumar/python-dotenv">python-dotenv</a> for environment variable management.</li>
</ul>

<h2>Contact</h2>
<p>Created by Siddharth Jain. You can reach out to me on <a href="https://www.linkedin.com/in/siddharth-jain-8b56a2321/">LinkedIn</a> or check out my other projects on <a href="https://github.com/S-AILAB">GitHub</a>.</p>

</body>
</html>
