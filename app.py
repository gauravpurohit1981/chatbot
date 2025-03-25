import os
import streamlit as st
import base64
from io import BytesIO
import tempfile
from PIL import Image
import time

from document_processor import (
    process_document,
    get_document_ids,
    delete_document,
    load_existing_documents,
)
from chat_engine import get_chat_response
from tabs import get_tab_content
from utils import get_background_image_style

# Set page config
st.set_page_config(
    page_title="Document-based Private Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_ids" not in st.session_state:
    st.session_state.document_ids = get_document_ids()
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "background_image" not in st.session_state:
    st.session_state.background_image = None
if "processing" not in st.session_state:
    st.session_state.processing = False
# Add interface mode (admin or user)
if "interface_mode" not in st.session_state:
    st.session_state.interface_mode = "user"  # Default to user mode
# Add authentication states
if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False
if "login_attempts" not in st.session_state:
    st.session_state.login_attempts = 0
if "show_login" not in st.session_state:
    st.session_state.show_login = False
    
# Admin credentials (in a real app, use proper authentication)
# For this demo, hardcoded credentials are used
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "secure_password123"

# Load vector store at startup
if st.session_state.vector_store is None:
    with st.spinner("Loading document database..."):
        st.session_state.vector_store = load_existing_documents()

# Set background image if available
if st.session_state.background_image:
    background_style = get_background_image_style(st.session_state.background_image)
    st.markdown(background_style, unsafe_allow_html=True)
else:
    # Use default background
    default_bg = "assets/default_background.svg"
    background_style = get_background_image_style(default_bg, is_local=True)
    st.markdown(background_style, unsafe_allow_html=True)

# Helper functions for authentication
def authenticate(username, password):
    """Simple authentication check"""
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        st.session_state.is_authenticated = True
        st.session_state.interface_mode = "admin"
        st.session_state.login_attempts = 0
        return True
    else:
        st.session_state.login_attempts += 1
        return False

def logout():
    """Log out user"""
    st.session_state.is_authenticated = False
    st.session_state.interface_mode = "user"
    st.session_state.show_login = False

# Sidebar for navigation, document management and settings
with st.sidebar:
    # Count documents
    document_count = len(st.session_state.document_ids)
    
    # Interface section
    st.title("Interface")
    
    # Admin access control
    if not st.session_state.is_authenticated:
        if st.button("Admin Login") or st.session_state.show_login:
            st.session_state.show_login = True
            with st.form("login_form"):
                st.subheader("Admin Login")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                login_button = st.form_submit_button("Log in")
                
                if login_button:
                    if authenticate(username, password):
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        if st.session_state.login_attempts >= 3:
                            st.error("Too many failed attempts. Please try again later.")
                            st.session_state.show_login = False
                        else:
                            st.error("Invalid credentials. Please try again.")
        
        # User is in normal user mode
        st.info("You are in user mode. Admin login required for document management.")
    else:
        # User is authenticated as admin
        st.success("Logged in as Administrator")
        if st.button("Logout"):
            logout()
            st.rerun()
    
    # Display different sidebar based on mode
    if st.session_state.interface_mode == "admin" and st.session_state.is_authenticated:
        st.title("Document Management")
        
        # Document upload
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )
        
        if uploaded_files:
            process_button = st.button("Process Documents")
            if process_button:
                st.session_state.processing = True
                progress_bar = st.progress(0)
                
                for i, uploaded_file in enumerate(uploaded_files):
                    # Save the uploaded file to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Process the document
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        try:
                            success = process_document(tmp_path, uploaded_file.name, st.session_state.vector_store)
                            if success:
                                st.success(f"Processed {uploaded_file.name}")
                            else:
                                st.error(f"Failed to process {uploaded_file.name}")
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        finally:
                            # Clean up the temporary file
                            os.unlink(tmp_path)
                    
                    # Update progress
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                
                # Refresh document IDs
                st.session_state.document_ids = get_document_ids()
                st.session_state.vector_store = load_existing_documents()
                st.session_state.processing = False
                st.success("All documents processed!")
                st.rerun()
        
        # Document management
        st.subheader("Manage Documents")
        st.write(f"Total documents: {document_count}/50")
        
        if document_count > 0:
            doc_to_delete = st.selectbox(
                "Select document to delete",
                options=st.session_state.document_ids,
                format_func=lambda x: x,
            )
            
            if st.button("Delete Selected Document"):
                with st.spinner("Deleting document..."):
                    success = delete_document(doc_to_delete, st.session_state.vector_store)
                    if success:
                        st.success(f"Deleted {doc_to_delete}")
                        # Refresh document IDs and vector store
                        st.session_state.document_ids = get_document_ids()
                        st.session_state.vector_store = load_existing_documents()
                        st.rerun()
                    else:
                        st.error(f"Failed to delete {doc_to_delete}")
        
        # Background image upload
        st.subheader("Appearance")
        new_bg_image = st.file_uploader(
            "Upload Background Image",
            type=["jpg", "jpeg", "png"],
            help="Upload a robotic style background image"
        )
        
        if new_bg_image:
            # Process and set as background
            try:
                img = Image.open(new_bg_image)
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                st.session_state.background_image = f"data:image/jpeg;base64,{img_str}"
                st.success("Background updated! Refreshing...")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    else:  # User mode
        # Simplified sidebar for users
        st.title("Document Information")
        st.write(f"Documents available: {document_count}")
        
        if document_count == 0:
            st.warning("No documents are available yet. Please contact your administrator to add documents.")
        else:
            st.info("This chatbot contains information from documents provided by administrators.")
            st.success("Ask specific questions in the chat to get answers from the document knowledge base.")
        
        # Chat help
        st.subheader("Chat Help")
        st.markdown("""
        - The chatbot only knows about the loaded documents
        - Ask specific questions for better results
        - Clear the chat using the button in the chat tab
        - Contact administrators if you need more information
        """)
        
        # Display document list
        if document_count > 0 and st.checkbox("Show available documents"):
            st.subheader("Available Documents")
            for doc_id in st.session_state.document_ids:
                st.write(f"• {doc_id.split('_')[0]}")

# Main content
# Different title and layout based on interface mode
if st.session_state.interface_mode == "admin":
    st.title("Document-based Private Chatbot - Admin Interface 🤖⚙️")
else:
    st.title("Document-based Private Chatbot 🤖")

# Select tabs based on interface mode
if st.session_state.interface_mode == "admin":
    # Full set of tabs for admin
    tab_titles = [
        "Chat",
        "Python Overview",
        "Secure by Design",
        "Security Team",
        "Privacy Team",
        "Project Overview",
        "Data Policy",
        "Technical Stack",
        "User Guidelines",
        "Compliance",
        "FAQ",
        "System Architecture",
        "Contact Information",
        "Progress Tracking",
        "Updates & Roadmap"
    ]
else:
    # Limited tabs for users, focusing on essential information
    tab_titles = [
        "Chat",
        "Project Overview",
        "User Guidelines",
        "FAQ",
        "Contact Information"
    ]

tabs = st.tabs(tab_titles)

# Chat Tab
with tabs[0]:
    if document_count == 0:
        if st.session_state.interface_mode == "admin":
            st.warning("Please upload documents to enable the chatbot.")
            st.info("Use the sidebar to upload PDF, DOCX, or TXT files, then click 'Process Documents' to start chatting.")
            # Add a sample question to demonstrate functionality
            st.markdown("### Sample Questions (once you upload documents)")
            st.markdown("""
            - What are the main topics covered in the documents?
            - Can you summarize the key points in the uploaded files?
            - What is the relationship between [topic A] and [topic B] in the documents?
            """)
        else:
            st.warning("No documents are available yet.")
            st.info("Please contact your administrator to add documents to the system.")
    else:
        # Chat controls - slightly different for admin vs user
        if st.session_state.interface_mode == "admin":
            col1, col2, col3 = st.columns([1, 5, 2])
            with col1:
                if st.button("Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()
            with col2:
                st.write(f"Documents available: {document_count}")
            with col3:
                st.write("Admin Mode: Full document control")
        else:
            # User mode has simpler interface
            col1, col2 = st.columns([1, 7])
            with col1:
                if st.button("Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()
            with col2:
                st.write("Ask questions about the document knowledge base")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input - slightly different placeholder based on mode
        if st.session_state.interface_mode == "admin":
            user_query = st.chat_input("Ask something about your documents...")
        else:
            user_query = st.chat_input("Ask a question about the documentation...")
        
        if user_query:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_query)
            
            # Get chatbot response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing documents..."):
                    if st.session_state.vector_store:
                        response = get_chat_response(user_query, st.session_state.vector_store)
                        st.markdown(response)
                        # Add assistant message to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    else:
                        st.error("Document database not loaded. Please refresh the page.")
                        
        # Help text at the bottom - different for admin vs user
        if st.session_state.chat_history:
            st.markdown("---")
            with st.expander("Chat Help"):
                if st.session_state.interface_mode == "admin":
                    st.markdown("""
                    - The chatbot only knows about the documents you've uploaded
                    - Ask specific questions for better results
                    - Clear the chat using the button above if the conversation gets too long
                    - Upload more documents if you need additional information
                    - Switch to User Mode to see the end-user experience
                    """)
                else:
                    st.markdown("""
                    - Ask specific questions about the documentation to get detailed answers
                    - The chatbot only has knowledge of the documents added by administrators
                    - For questions not answered by the bot, contact the administrators
                    """)

# Other tabs
for i in range(1, len(tab_titles)):
    with tabs[i]:
        tab_content = get_tab_content(tab_titles[i])
        st.markdown(tab_content)

# Footer
st.markdown("---")
if st.session_state.interface_mode == "admin":
    st.caption("Document-based Private Chatbot (Admin Interface) • Built with Streamlit, LangChain, and scikit-learn for 100% offline use")
else:
    st.caption("Document-based Private Chatbot • Your centralized document knowledge base")
