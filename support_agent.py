import streamlit as st
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Tikitly Support",
    page_icon="◆",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Professional minimal CSS
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #fafaf9;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header */
    .custom-header {
        background-color: #ffffff;
        padding: 1.5rem 2rem;
        border-bottom: 1px solid #e7e5e4;
        margin: -1rem -1rem 2rem -1rem;
    }
    
    .custom-header h1 {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1c1917;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .custom-header p {
        font-size: 0.875rem;
        color: #78716c;
        margin: 0.25rem 0 0 0;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: #ffffff;
        border: 1px solid #e7e5e4;
        border-radius: 0.5rem;
        padding: 1.25rem;
        margin: 0.75rem 0;
    }
    
    .stChatMessage[data-testid="user-message"] {
        background-color: #fafaf9;
        border-color: #d6d3d1;
    }
    
    /* Chat input */
    .stChatInput {
        border-radius: 0.5rem;
    }
    
    .stChatInput > div {
        background-color: #ffffff;
        border: 1px solid #d6d3d1;
        border-radius: 0.5rem;
    }
    
    .stChatInput input {
        color: #1c1917;
    }
    
    .stChatInput input::placeholder {
        color: #a8a29e;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #1c1917;
        color: #ffffff;
        border: none;
        border-radius: 0.375rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        background-color: #292524;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Loading state */
    .stSpinner > div {
        border-color: #1c1917 transparent transparent transparent;
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 9999px;
        font-size: 0.75rem;
        color: #15803d;
        font-weight: 500;
    }
    
    .status-dot {
        width: 6px;
        height: 6px;
        background-color: #22c55e;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    
    .error-box {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #991b1b;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'load_error' not in st.session_state:
    st.session_state.load_error = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Path to pre-generated vector store
VECTOR_STORE_PATH = "./vector_store"

@st.cache_resource
def load_vector_store():
    """Load pre-generated Ollama embeddings from disk (cached, runs once)"""
    try:
        if not os.path.exists(VECTOR_STORE_PATH):
            return None, "Vector store not found. Please run generate_embeddings.py first."
        
        # Create embeddings object (same model used to generate)
        embeddings = SentenceTransformerEmbeddings(
            model_name="all-mpnet-base-v2"
        )
        
        # Load the pre-generated vector store
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        return vector_store, None
        
    except Exception as e:
        return None, f"Error loading vector store: {str(e)}"

def create_advanced_retrieval_chain(vector_store):
    """Create RAG chain with Ollama embeddings + Gemini LLM"""
    
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 15,
            "lambda_mult": 0.6
        }
    )
    
    # Use Gemini for fast, high-quality responses
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0.1
    )
    
    prompt = PromptTemplate(
        template="""You are the Tikitly Support Agent, assisting event organizers with the Tikitly platform.

KNOWLEDGE BASE CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
1. Answer ONLY using information from the provided Tikitly documentation
2. If information is not available, respond with: "I don't have that information in my current knowledge base. Please contact Tikitly support for assistance."
3. Never guess or assume platform features
4. Provide clear, step-by-step instructions when explaining workflows
5. Be professional and concise
6. When referencing UI elements, be specific about location and action

Provide a direct, helpful answer:""",
        input_variables=['context', 'question']
    )
    
    def format_docs(retrieved_docs):
        if not retrieved_docs:
            return "No relevant information found."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get('source', 'Documentation')
            source_info = f"{os.path.basename(source)}"
            context_parts.append(f"[Source {i} - {source_info}]:\n{doc.page_content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })
    
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser
    
    return main_chain

# Quick action suggestions
QUICK_ACTIONS = [
    "How do I create a new event?",
    "How do I add and configure ticket types?",
    "What are add-ons and how do I create them?",
    "Explain the different commission models",
    "How can I issue complimentary tickets?",
    "How do I check my event metrics?",
]

def main():
    # Auto-load vector store on first run
    if st.session_state.vector_store is None and st.session_state.load_error is None:
        vector_store, error = load_vector_store()
        if error:
            st.session_state.load_error = error
        else:
            st.session_state.vector_store = vector_store
    
    # Custom header
    st.markdown("""
    <div class='custom-header'>
        <h1>Tikitly Support</h1>
        <p>Get help with event creation, tickets, and platform features</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show error if loading failed
    if st.session_state.load_error:
        st.markdown(f"""
        <div class='error-box'>
            <strong>Setup Required:</strong><br>
            {st.session_state.load_error}<br><br>
            <strong>Steps to fix:</strong><br>
            1. Make sure Ollama is running: <code>ollama serve</code><br>
            2. Pull the embedding model: <code>ollama pull nomic-embed-text</code><br>
            3. Run <code>python generate_embeddings.py</code> once<br>
            4. Restart this app
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Knowledge base status and controls
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col2:
        st.markdown('<div class="status-indicator"><span class="status-dot"></span>Ready</div>', unsafe_allow_html=True)
    
    with col3:
        if len(st.session_state.chat_history) > 0:
            if st.button("Clear chat", key="clear"):
                st.session_state.chat_history = []
                st.rerun()
    
    # Quick actions (only show when no chat history)
    if not st.session_state.chat_history:
        st.markdown("<p style='color: #78716c; font-size: 0.875rem; margin: 1.5rem 0 0.75rem 0;'>Common questions:</p>", unsafe_allow_html=True)
        
        cols = st.columns(2)
        for idx, question in enumerate(QUICK_ACTIONS):
            with cols[idx % 2]:
                if st.button(question, key=f"quick_{idx}", use_container_width=True):
                    st.session_state.pending_question = question
                    st.rerun()
        
        st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
    
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])
    
    # Handle pending question from quick action
    if hasattr(st.session_state, 'pending_question'):
        question = st.session_state.pending_question
        delattr(st.session_state, 'pending_question')
        
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    chain = create_advanced_retrieval_chain(st.session_state.vector_store)
                    answer = chain.invoke(question)
                    st.markdown(answer)
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer
                    })
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Chat input
    question = st.chat_input("Ask a question about Tikitly...")
    
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    chain = create_advanced_retrieval_chain(st.session_state.vector_store)
                    answer = chain.invoke(question)
                    st.markdown(answer)
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer
                    })
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()