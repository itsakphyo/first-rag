import streamlit as st
from first_rag.db_helpers import generate_data_store
from first_rag.chat_helpers import generate_response
import shutil
import os

st.title('Chat with your PDF')
st.sidebar.title("Vector Database Configuration")

uploaded_files = st.sidebar.file_uploader("Upload Documents", accept_multiple_files=True, type=["txt", "pdf"])

db_name = st.sidebar.text_input("Database Name", value="default_db")
chunk_size = st.sidebar.text_input("Chunk Size", value="500")
chunk_overlap = st.sidebar.text_input("Chunk Overlap", value="50")

if 'db_path' not in st.session_state:
    st.session_state.db_path = None

if st.sidebar.button("Create Vector Database"):
    st.session_state.db_path = f"data/database/{db_name}"
    db_path = st.session_state.db_path

    if not uploaded_files:
        st.sidebar.error("Please upload a file to create the database.")
    elif not chunk_size or not chunk_overlap:
        st.sidebar.error("Please provide valid chunk size and overlap values.")
    elif os.path.exists(db_path):
        st.sidebar.error(f"Database '{db_name}' already exists. Please choose a different name.")
    else:
        try:
            chunk_size = int(chunk_size)
            chunk_overlap = int(chunk_overlap)
        except ValueError:
            st.sidebar.error("Please provide valid numeric values for chunk size and overlap.")
            chunk_size = chunk_overlap = None

        if chunk_size and chunk_overlap:
            shutil.rmtree("data/raw", ignore_errors=True)
            os.makedirs("data/raw", exist_ok=True)

            for uploaded_file in uploaded_files:
                file_path = f"data/raw/{uploaded_file.name}"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            os.makedirs(db_path, exist_ok=True)
            for file_name in os.listdir("data/raw"):
                file_path = f"data/raw/{file_name}"
                generate_data_store(file_path, chunk_size, chunk_overlap, db_path)

            st.sidebar.success(f"Vector database '{db_name}' created successfully!")

old_dbs = [db for db in os.listdir("data/database")]
old_db_name = st.sidebar.selectbox("Use Old Vector Database", old_dbs) 

if st.sidebar.button("Select Vector Database"):
    st.session_state.db_path = f"data/database/{old_db_name}"
    st.session_state['messages'] = []
    st.sidebar.success(f"Using old vector database '{old_db_name}'.")

if st.sidebar.button("Stop Using Vector Database"):
    st.session_state.db_path = None
    st.session_state['messages'] = []
    st.sidebar.success(f"Stopped using vector database.")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if prompt := st.chat_input("Enter your query"):
    st.session_state['messages'].append({"role": "user", "content": prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    response_stream = generate_response(prompt, db_path=st.session_state.db_path, history=st.session_state['messages'])

    assistant_message = st.chat_message('assistant')
    placeholder = assistant_message.empty()
    response_text = ""

    for partial_response in response_stream:
        response_text += partial_response  
        placeholder.markdown(response_text)  

    st.session_state['messages'].append({"role": "assistant", "content": response_text})
