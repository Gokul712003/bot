import streamlit as st
from chatbot import chat_with_memory, process_pdf, chat_with_pdf, write_assignment
import tempfile
import os
from langchain.schema.messages import HumanMessage,AIMessage
from docx import Document
import io
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="AI Chatbot with Multiple Search Tools",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state variables
if "chat_messages" not in st.session_state:  # for tab 1
    st.session_state.chat_messages = []
if "pdf_messages" not in st.session_state:   # for tab 2
    st.session_state.pdf_messages = []
if "chat_state" not in st.session_state:
    st.session_state.chat_state = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_chat_history" not in st.session_state:
    st.session_state.pdf_chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar with capabilities
with st.sidebar:
    st.title("🛠️ Available Tools")
    
    # PDF upload section
    # st.markdown("### 📄 PDF Upload")
    # uploaded_pdf = st.file_uploader("Upload a PDF for context", type=['pdf'])
    
    # if uploaded_pdf is not None:
    #     with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
    #         tmp_file.write(uploaded_pdf.getvalue())
    #         tmp_file_path = tmp_file.name
        
    #     with st.spinner('Processing PDF...'):
    #         try:
    #             st.session_state.vectorstore = process_pdf(tmp_file_path)
    #             st.success("PDF processed successfully!")
    #         except Exception as e:
    #             st.error(f"Error processing PDF: {str(e)}")
    #         finally:
    #             os.unlink(tmp_file_path)

    # Tool information
    st.markdown("""
    ### 🎥 YouTube Search
    Ask for videos using keywords like:
    - "Show me a video about..."
    - "Find a tutorial on..."
    - "I want to watch..."
    
    ### 📚 ArXiv Research
    Find academic papers using:
    - "Find research papers about..."
    - "Show me studies on..."
    - "Latest publications in..."
    
    ### 🔍 DuckDuckGo Search
    Get general information with:
    - "What is..."
    - "Tell me about..."
    - "Find information on..."
    
    ### 🔬 Tavily Search
    Get detailed analysis using:
    - "Analyze..."
    - "Explain in detail..."
    - "Comprehensive information about..."
    """)

# Function to process user input
def process_input():
    if st.session_state.user_input and st.session_state.user_input.strip():
        user_message = st.session_state.user_input
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_message})
        
        # Get bot response
        response, new_state = chat_with_memory(user_message, st.session_state.chat_state)
        st.session_state.chat_state = new_state
        
        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Clear the input
        st.session_state.user_input = ""

# Add custom CSS for chat styling
st.markdown("""
    <style>
    .user-message {
        background-color: #DCF8C6;
        padding: 15px;
        border-radius: 15px;
        margin: 5px 20px 5px 70px;
        display: inline-block;
        max-width: 70%;
        float: right;
        clear: both;
        word-wrap: break-word;
        color: #000000;  /* Black text color */
    }
    .assistant-message {
        background-color: #E8E8E8;
        padding: 15px;
        border-radius: 15px;
        margin: 5px 70px 5px 20px;
        display: inline-block;
        max-width: 70%;
        float: left;
        clear: both;
        word-wrap: break-word;
        color: #000000;  /* Black text color */
    }
    .chat-container {
        width: 100%;
        overflow: hidden;
        padding: 10px;
    }
    .stMarkdown {
        max-width: 100%;
    }
    /* Add dark theme compatibility */
    @media (prefers-color-scheme: dark) {
        .user-message, .assistant-message {
            color: #000000;  /* Keep text black even in dark mode */
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Main chat interface
def send_feedback_email(name: str, email: str, feedback: str) -> bool:
    # Email configuration
    smtp_server = "smtp.gmail.com"
    port = 587
    sender_email = os.getenv("GMAIL_USER")  # Your Gmail
    password = os.getenv("GMAIL_PASSWORD")   # App password from .env
    receiver_email = "gokulaprasath2003@gmail.com"

    if not password:
        st.error("Email configuration is missing. Please contact the administrator.")
        return False

    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = f"AI Assistant Feedback from {name}"

        body = f"""
        New feedback received:
        
        Name: {name}
        Email: {email}
        
        Feedback:
        {feedback}
        """
        msg.attach(MIMEText(body, 'plain'))

        # Create secure SSL/TLS connection
        server = smtplib.SMTP(smtp_server, port)
        server.starttls()
        
        # Login with your credentials
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        return True

    except Exception as e:
        print(f"Error sending email: {str(e)}")
        st.error(f"Error sending email: {str(e)}")
        return False

st.title("🤖 AI Assistant with Search Tools")

# After the title, add tab selection
tab1, tab2, tab3, tab4 = st.tabs(["Chat with Tools", "PDF Chat", "Assignment Writer", "Feedback"])

with tab1:
    # Display chat messages for tab 1
    for message in st.session_state.chat_messages:
        if message["role"] == "user":
            with st.container():
                st.markdown(f"""
                    <div class="chat-container">
                        <div class="user-message">
                            {message["content"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            with st.container():
                st.markdown(f"""
                    <div class="chat-container">
                        <div class="assistant-message">
                            {message["content"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # Handle user input for tab 1
    if prompt := st.chat_input("What would you like to know?", key="chat_input"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Thinking..."):
            response, new_state = chat_with_memory(
                prompt, 
                st.session_state.chat_state
            )
            
            st.session_state.chat_state = new_state
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
        
        st.rerun()

    # Auto-scroll to bottom (add this at the end of your script)
    st.markdown("""
        <script>
            var elements = window.parent.document.querySelectorAll('.stMarkdown');
            var lastElement = elements[elements.length - 1];
            if (lastElement) {
                lastElement.scrollIntoView();
            }
        </script>
        """, unsafe_allow_html=True)

with tab2:
    st.title("📄 Chat with PDF")
    
    uploaded_pdf = st.file_uploader("Upload a PDF for context", type=['pdf'])
    
    if uploaded_pdf is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_pdf.getvalue())
            tmp_file_path = tmp_file.name
        
        with st.spinner('Processing PDF...'):
            try:
                st.session_state.vectorstore = process_pdf(tmp_file_path)
                st.success("PDF processed successfully!")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
            finally:
                os.unlink(tmp_file_path)
    
    # Display PDF chat messages
    for message in st.session_state.pdf_messages:
        if message["role"] == "user":
            with st.container():
                st.markdown(f"""
                    <div class="chat-container">
                        <div class="user-message">
                            {message["content"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            with st.container():
                st.markdown(f"""
                    <div class="chat-container">
                        <div class="assistant-message">
                            {message["content"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # PDF chat interface
    if st.session_state.vectorstore is not None:
        if pdf_prompt := st.chat_input("Ask questions about your PDF", key="pdf_input"):
            st.session_state.pdf_messages.append({"role": "user", "content": pdf_prompt})
            
            with st.spinner("Searching PDF..."):
                docs = st.session_state.vectorstore.similarity_search(pdf_prompt)
                context = "\n".join([doc.page_content for doc in docs])
                
                # Convert message history to BaseMessage format
                chat_history = [
                    HumanMessage(content=m["content"]) if m["role"] == "user" 
                    else AIMessage(content=m["content"])
                    for m in st.session_state.pdf_messages[:-1]  # Exclude the current message
                ]
                
                response = chat_with_pdf(pdf_prompt, context, chat_history)
                st.session_state.pdf_messages.append({"role": "assistant", "content": response})
            
            st.rerun() 

with tab3:
    st.title("📝 Assignment Writer")
    
    topic = st.text_input("Enter Assignment Topic and Requirements")
    requirements = st.text_area("Enter your name and Reg Number (optional)")
    
    if st.button("Generate Assignment") and topic:
        with st.spinner("Writing Assignment..."):
            try:
                # Generate content using local Blank.docx
                content, doc = write_assignment(topic, requirements)
                
                # Display preview
                st.markdown("### Preview:")
                st.markdown(content)
                
                # Save document to bytes
                doc_bytes = io.BytesIO()
                doc.save(doc_bytes)
                doc_bytes.seek(0)
                
                # Download button for Word document
                st.download_button(
                    label="Download as Word Document",
                    data=doc_bytes,
                    file_name=f"{topic.lower().replace(' ', '_')}_assignment.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
                
            except Exception as e:
                st.error(f"Error generating assignment: {str(e)}") 

with tab4:
    st.title("💌 Feedback")
    
    col1, col2 = st.columns(2)
    with col1:
        feedback_name = st.text_input("Your Name")
    with col2:
        feedback_email = st.text_input("Your Email")
    
    feedback_content = st.text_area("Your Feedback")
    
    if st.button("Submit Feedback"):
        if not feedback_name or not feedback_email or not feedback_content:
            st.warning("Please fill in all fields.")
        elif '@' not in feedback_email:
            st.warning("Please enter a valid email address.")
        else:
            with st.spinner("Sending feedback..."):
                if send_feedback_email(feedback_name, feedback_email, feedback_content):
                    st.success("Thank you for your feedback! We'll get back to you soon.")
                    # Clear the form using the new rerun method
                    st.rerun()

# Add contact information at the bottom of the page
st.markdown(
    """
    <div style='position: fixed; bottom: 10px; right: 10px; padding: 10px; 
    background-color: rgba(255,255,255,0.1); border-radius: 5px;'>
    Contact: <a href='mailto:gokulaprasath.s2024@vitstudent.ac.in'>gokulaprasath.s2024@vitstudent.ac.in</a>
    </div>
    """, 
    unsafe_allow_html=True
) 