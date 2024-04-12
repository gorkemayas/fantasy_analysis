import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
import openai
import yt_dlp
import os
import tempfile

def download_youtube_audio(youtube_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': tempfile.mktemp(dir='.', suffix='.wav')  # Temporary file
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
        info_dict = ydl.extract_info(youtube_url, download=False)
        return ydl.prepare_filename(info_dict)

def speech_to_text(audio_file_path):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        st.error("Google Speech Recognition could not understand the audio")
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
    except Exception as e:
        st.error(f"Error processing the audio file: {e}")
    finally:
        os.remove(audio_file_path)  # Remove the temp file
    return None

def use_openai_api(text, api_key):
    openai.api_key = api_key
    prompt_text = f"""
    Analyze the following text to reveal hidden messages and emotional undertones:
    "{text}"

    Steps:
    1. Identify strong feeling words, even in mundane contexts.
    2. Note all metaphors and similes, especially recurring ones.
    3. Record all family-related terms like "mother", "father", "children".
    4. Ignore negatives to focus on core subjects, e.g., interpret "we don’t want war" as "we want peace".
    5. Reconstruct sentences using the recorded words and phrases to reveal hidden messages and emotional undertones.
    """

    response = openai.Completion.create(
        engine="gpt-3.5-turbo-0125",
        prompt=prompt_text,
        max_tokens=500
    )

    return response.choices[0].text.strip()

def main():
    st.title('Audio to Text Analysis Using AI')
    api_key = st.text_input("Enter your OpenAI API Key:", type="password")

    input_method = st.radio("Choose input method:", options=['Upload Audio', 'YouTube Link', 'Paste Text'])

    if input_method == 'Upload Audio':
        audio_file = st.file_uploader("Upload your audio file", type=['mp3', 'wav', 'ogg', 'mp4', 'm4a'])
        if audio_file and api_key:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                audio = AudioSegment.from_file(audio_file)
                audio.export(tmp.name, format="wav")
                text = speech_to_text(tmp.name)
                if text:
                    result = use_openai_api(text, api_key)
                    st.write("Analysis:\n", result)

    elif input_method == 'YouTube Link':
        youtube_url = st.text_input("Enter YouTube URL:")
        if youtube_url and st.button("Download and Process"):
            audio_file_path = download_youtube_audio(youtube_url)
            text = speech_to_text(audio_file_path)
            if text:
                result = use_openai_api(text, api_key)
                st.write("Analysis:\n", result)

    elif input_method == 'Paste Text':
        user_text = st.text_area("Paste your text here:")
        if st.button("Analyze Text"):
            if user_text and api_key:
                result = use_openai_api(user_text, api_key)
                st.write("Analysis:\n", result)

if __name__ == "__main__":
    main()
