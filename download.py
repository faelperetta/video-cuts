from yt_dlp import YoutubeDL

url = "https://www.youtube.com/watch?v=mfv0V1SxbNA"

ydl_opts = {
    "format": "bestvideo+bestaudio/best",
    "outtmpl": "input.%(ext)s",
}

with YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])