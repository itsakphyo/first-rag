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
