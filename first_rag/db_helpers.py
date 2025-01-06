from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from first_rag.config import settings

def load_documents(data_path):
    file_extension = data_path.split('.')[-1].lower()

    if file_extension == 'pdf':
        document_loader = PyPDFLoader(data_path)
        return document_loader.load()
    elif file_extension == 'txt':
        with open(data_path, "r", encoding="utf-8") as file:
            text = file.read()
            return [Document(page_content=text, metadata={})] 
    else:
        raise ValueError("Unsupported file format. Please provide a PDF or TXT file.")
    
def split_text(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks, chroma_path):
    batch_size = 100
    total_chunks = len(chunks)
    
    db = Chroma.from_documents(
        chunks[:batch_size],
        OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY),
        persist_directory=chroma_path
    )
    
    for i in range(batch_size, total_chunks, batch_size):
        batch = chunks[i:i+batch_size]
        db.add_documents(batch)
    
    print(f"Saved {total_chunks} chunks to {chroma_path}.")

def generate_data_store(data_path, chunk_size, chunk_overlap, chroma_path):
    documents = load_documents(data_path)
    chunks = split_text(documents, chunk_size, chunk_overlap)
    save_to_chroma(chunks, chroma_path)


# def get_yt_video_transcribed(url):
#     yt = pt.YouTube(url)
#     yt.streams.filter(only_audio=True).first().download(filename="yt_audio.mp3", output_path=path)
#     model = whisper.load_model("base")
#     transcribed_text = model.transcribe(path)




# import pytube as pt
# import whisper

# def download_video(url, path):
#     yt = pt.YouTube(url)
#     yt.streams.filter(progressive=True, file_extension='mp4').first().download(filename="yt_video.mp4", output_path=path)

# def download_audio(url, path):
#     yt = pt.YouTube(url)
#     yt.streams.filter(only_audio=True).first().download(filename="yt_audio.mp3", output_path=path)


# path = "yt_audio.mp3"

# model = whisper.load_model("base")
# transcribed_text = model.transcribe(path)