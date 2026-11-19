import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

if not api_key:
    print("API Key tidak ditemukan!")
else:
    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=model,
            contents="Perkenalkan SmartTour AI dalam satu kalimat."
        )

        print("Koneksi Gemini berhasil!")
        print(response.text)

    except Exception as e:
        print("Koneksi gagal:", e)