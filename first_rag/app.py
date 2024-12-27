import streamlit as st
from first_rag.utils import generate_response, generate_data_store
import shutil
import os

# Set up the page title
st.title('Chat with me')

# Sidebar for Vector Database Configuration
st.sidebar.title("Vector Database Configuration")

# File uploader for document files
uploaded_files = st.sidebar.file_uploader("Upload Documents", accept_multiple_files=True, type=["txt", "pdf"])

# Inputs for database configuration
db_name = st.sidebar.text_input("Database Name", value="default_db")
chunk_size = st.sidebar.text_input("Chunk Size", value="500")
chunk_overlap = st.sidebar.text_input("Chunk Overlap", value="50")

# Button for creating the vector database
if st.sidebar.button("Create Vector Database"):
    db_path = f"data/database/{db_name}"

    # Validate uploaded files and input values
    if not uploaded_files:
        st.sidebar.error("Please upload a file to create the database.")
    elif not chunk_size or not chunk_overlap:
        st.sidebar.error("Please provide valid chunk size and overlap values.")
    elif os.path.exists(db_path):
        st.sidebar.error(f"Database '{db_name}' already exists. Please choose a different name.")
    else:
        try:
            # Ensure chunk size and overlap are integers
            chunk_size = int(chunk_size)
            chunk_overlap = int(chunk_overlap)
        except ValueError:
            st.sidebar.error("Please provide valid numeric values for chunk size and overlap.")
            chunk_size = chunk_overlap = None

        if chunk_size and chunk_overlap:
            shutil.rmtree("data/raw", ignore_errors=True)
            os.makedirs("data/raw", exist_ok=True)

            # Save uploaded files to the raw directory
            for uploaded_file in uploaded_files:
                file_path = f"data/raw/{uploaded_file.name}"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            # Create the database and store vector data
            os.makedirs(db_path, exist_ok=True)
            for file_name in os.listdir("data/raw"):
                file_path = f"data/raw/{file_name}"
                generate_data_store(file_path, chunk_size, chunk_overlap, db_path)

            st.sidebar.success(f"Vector database '{db_name}' created successfully!")

# Initialize session state for chat messages
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display previous messages in the chat interface
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Handle user input and generate assistant response
if prompt := st.chat_input("Enter your query"):
    # Save user input
    st.session_state['messages'].append({"role": "user", "content": prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    # Get response from the custom function
    response = generate_response(prompt)
    with st.chat_message('assistant'):
        st.markdown(response)

    # Save assistant's response
    st.session_state['messages'].append({"role": "assistant", "content": response})
