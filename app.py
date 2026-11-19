import time
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. KONFIGURASI DASAR
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "wisata_banten.csv"
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH, encoding="utf-8-sig")

try:
    GEMINI_API_KEY =(
        os.getenv("GEMINI_API_KEY") 
        or st.secrets.get("GEMINI_API_KEY", "")
        ).strip()
                  
    GEMINI_MODEL = (
        os.getenv("GEMINI_MODEL") or 
        st.secrets.get("GEMINI_MODEL","gemini-3.8-flash")
        ).strip()

except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

NAMA_PENGEMBANG = "Rangga Rakhazona Pamungkas"

st.set_page_config(
    page_title="SmartTour AI | Wisata Banten",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. STYLE / UI
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --st-primary: #087e75;
        --st-deep: #073b3a;
        --st-mint: #dff5ed;
        --st-ink: #173b3a;
        --st-muted: #66817d;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
          radial-gradient(circle at 88% 3%, rgba(8,126,117,.08), transparent 24%),
          linear-gradient(180deg, #f7fbf9 0%, #f2f8f5 100%);
    }

    [data-testid="stHeader"] {
        background: rgba(247,251,249,.82);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #073b3a 0%, #0b5550 100%);
        border-right: 1px solid rgba(255,255,255,.08);
    }

    [data-testid="stSidebar"] * {
        color: #effaf6 !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 8px 10px;
        border-radius: 10px;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(255,255,255,.10);
    }

    .brand-lockup {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 2px 20px 2px;
    }

    .brand-icon {
        width: 48px;
        height: 48px;
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 25px;
        background: linear-gradient(135deg, #a7f3d0, #2dd4bf);
        box-shadow: 0 8px 22px rgba(0,0,0,.16);
    }

    .brand-name {
        font-family: 'Manrope', sans-serif;
        font-size: 21px;
        line-height: 1.1;
        font-weight: 800;
        color: white;
        letter-spacing: -.6px;
    }

    .brand-sub {
        font-size: 11px;
        color: #b8ded4;
        margin-top: 4px;
        letter-spacing: .7px;
        text-transform: uppercase;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 30px 32px;
        border-radius: 24px;
        color: white;
        background: linear-gradient(115deg, #073b3a 0%, #087e75 58%, #20a58e 100%);
        box-shadow: 0 16px 38px rgba(7,59,58,.15);
        margin-bottom: 22px;
    }

    .hero:after {
        content: "🌴";
        position: absolute;
        right: 34px;
        bottom: -26px;
        font-size: 118px;
        opacity: .16;
        transform: rotate(-12deg);
    }

    .hero-eyebrow {
        color: #b9f3df;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.8px;
        font-weight: 700;
        margin-bottom: 9px;
    }

    .hero h1 {
        color: white;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(27px, 3vw, 39px);
        font-weight: 800;
        line-height: 1.15;
        margin: 0 0 10px 0;
    }

    .hero p {
        color: #e0f5ed;
        max-width: 650px;
        font-size: 15px;
        margin: 0;
        line-height: 1.65;
    }

    .section-kicker {
        color: #087e75;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .section-heading {
        font-family: 'Manrope', sans-serif;
        color: #173b3a;
        font-size: 23px;
        font-weight: 800;
        margin: 0 0 14px 0;
    }

    .metric-card {
        background: white;
        border: 1px solid #e2eee8;
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 6px 18px rgba(22,70,61,.045);
        min-height: 112px;
    }

    .metric-label {
        color: #66817d;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #073b3a;
        font-family: 'Manrope', sans-serif;
        font-size: 29px;
        font-weight: 800;
        line-height: 1.1;
    }

    .metric-note {
        color: #8ba29c;
        font-size: 11px;
        margin-top: 7px;
    }

    .soft-card {
        background: white;
        border: 1px solid #e2eee8;
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 6px 18px rgba(22,70,61,.04);
        margin-bottom: 12px;
    }

    .destination-title {
        color: #073b3a;
        font-family: 'Manrope', sans-serif;
        font-size: 18px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .destination-meta {
        color: #66817d;
        font-size: 12px;
        margin-bottom: 9px;
    }

    .pill {
        display: inline-block;
        padding: 4px 10px;
        margin: 2px 4px 2px 0;
        border-radius: 30px;
        background: #e4f6ef;
        color: #087e75;
        font-size: 11px;
        font-weight: 700;
    }

    .profile-card {
        border: 1px solid rgba(255,255,255,.17);
        background: rgba(255,255,255,.08);
        border-radius: 15px;
        padding: 13px;
        margin-top: 18px;
    }

    .profile-name {
        font-size: 13px;
        font-weight: 700;
        color: white;
        line-height: 1.4;
    }

    .profile-caption {
        font-size: 10px;
        color: #b8ded4;
        margin-top: 3px;
    }

    .stButton > button {
        border-radius: 11px;
        border: 1px solid #087e75;
        background: #087e75;
        color: white;
        font-weight: 700;
        transition: all .2s ease;
    }

    .stButton > button:hover {
        background: #06665f;
        border-color: #06665f;
        color: white;
        transform: translateY(-1px);
    }

    [data-testid="stChatMessage"] {
        border: 1px solid #e3eee9;
        border-radius: 16px;
        padding: 12px 14px;
        background: white;
    }

    [data-testid="stChatInput"] {
        border-color: #cfe3dc;
        border-radius: 15px;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 11px;
    }

    .footer {
        color: #829993;
        text-align: center;
        font-size: 11px;
        padding: 18px 0 4px 0;
    }

    @media (max-width: 700px) {
        .hero { padding: 24px 22px; }
        .hero:after { right: 8px; font-size: 78px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. MEMBACA DATASET
# =========================================================
@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}. Pastikan file berada di folder data."
        )

    data = pd.read_csv(path)
    required_columns = [
        "id",
        "nama_wisata",
        "lokasi",
        "kategori",
        "deskripsi",
        "aktivitas",
        "fasilitas",
        "keywords",
    ]

    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Kolom dataset belum lengkap: {missing}")

    return data.fillna("")


try:
    df = load_data(str(DATA_PATH))
except Exception as error:
    st.error(f"Gagal membaca dataset: {error}")
    st.stop()


# =========================================================
# 4. SISTEM REKOMENDASI TF-IDF
# =========================================================
df["fitur"] = (
    df["nama_wisata"].astype(str) + " "
    + df["lokasi"].astype(str) + " "
    + df["kategori"].astype(str) + " "
    + df["deskripsi"].astype(str) + " "
    + df["aktivitas"].astype(str) + " "
    + df["fasilitas"].astype(str) + " "
    + df["keywords"].astype(str)
)

vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df["fitur"])


def rekomendasi_wisata(query: str, jumlah: int = 5) -> pd.DataFrame:
    query = (query or "").strip()
    if not query:
        return df.head(0).copy()

    query_vector = vectorizer.transform([query])
    similarity = cosine_similarity(query_vector, tfidf_matrix).flatten()

    hasil = df.copy()
    hasil["skor_kemiripan"] = similarity
    hasil = hasil[hasil["skor_kemiripan"] > 0]
    hasil = hasil.sort_values("skor_kemiripan", ascending=False)
    return hasil.head(jumlah)


def format_konteks(data: pd.DataFrame) -> str:
    if data.empty:
        return "Tidak ditemukan destinasi yang relevan dalam dataset."
    kolom = [
        "nama_wisata",
        "lokasi",
        "kategori",
        "deskripsi",
        "aktivitas",
        "fasilitas",
    ]
    return data[kolom].to_string(index=False)


# =========================================================
# 5. INTEGRASI GEMINI
# =========================================================
def buat_jawaban_ai(
    pertanyaan: str,
    konteks: str,
    riwayat: list | None = None,
) -> str:
    if not GEMINI_API_KEY:
        return (
            "API Key Gemini belum ditemukan. Periksa variabel "
            "GEMINI_API_KEY pada file .env."
        )

    instruksi = """
Kamu adalah SmartTour AI, asisten virtual pariwisata Provinsi Banten.
Gunakan Bahasa Indonesia yang ramah, jelas, dan membantu.
Prioritaskan informasi yang tersedia pada DATA REFERENSI.
Jangan mengarang harga tiket, jam operasional, rating, alamat rinci,
nomor telepon, kondisi jalan, atau informasi transportasi.
Jika informasi tidak tersedia dalam referensi, sampaikan secara jujur.
Jika pengguna meminta rekomendasi, jelaskan alasan singkat berdasarkan
lokasi, kategori, aktivitas, atau deskripsi yang tersedia.
Jawab secara terstruktur dan tidak terlalu panjang.
"""

    riwayat_teks = ""
    if riwayat:
        for pesan in riwayat[-8:]:
            role = "Pengguna" if pesan.get("role") == "user" else "SmartTour AI"
            content = str(pesan.get("content", "")).strip()
            if content:
                riwayat_teks += f"{role}: {content}\n"

    prompt = f"""
{instruksi}

RIWAYAT PERCAKAPAN:
{riwayat_teks or "Belum ada riwayat percakapan."}

DATA REFERENSI WISATA BANTEN:
{konteks}

PERTANYAAN TERBARU:
{pertanyaan}

Jawab pertanyaan terbaru dengan mempertimbangkan riwayat dan data referensi.
"""

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        jawaban = getattr(response, "text", None)
        if jawaban and jawaban.strip():
            return jawaban.strip()

        return (
            "Gemini belum memberikan teks jawaban. "
            "Silakan coba pertanyaan yang lebih sederhana."
        )

    except Exception as error:
        pesan_error = str(error)
        print(f"Gemini API error: {repr(error)}", flush=True)

        if "429" in pesan_error or "RESOURCE_EXHAUSTED" in pesan_error:
            return (
                "Kuota atau batas permintaan Gemini sedang tercapai. "
                "Periksa kuota project di Google AI Studio, lalu coba lagi nanti."
            )
        if "404" in pesan_error or "NOT_FOUND" in pesan_error:
            return (
                f"Model Gemini '{GEMINI_MODEL}' tidak ditemukan atau tidak tersedia "
                "untuk API Key ini. Periksa nilai GEMINI_MODEL pada file .env."
            )
        if "401" in pesan_error or "403" in pesan_error or "PERMISSION_DENIED" in pesan_error:
            return (
                "Gemini menolak akses API. Periksa API Key, project, dan izin "
                "Gemini API di Google AI Studio."
            )
        if "timeout" in pesan_error.lower():
            return (
                "Permintaan ke Gemini melewati batas waktu. Periksa koneksi internet "
                "dan coba kembali."
            )

        return f"Terjadi kesalahan saat menghubungkan Gemini API: {pesan_error}"


# =========================================================
# 6. SIDEBAR / NAVIGASI
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div class="brand-lockup">
            <div class="brand-icon">🌴</div>
            <div>
                <div class="brand-name">SmartTour AI</div>
                <div class="brand-sub">Banten Tourism Assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-size:11px;color:#b8ded4;font-weight:700;"
        "letter-spacing:1px;margin:4px 0 8px 4px;'>MENU UTAMA</div>",
        unsafe_allow_html=True,
    )

    menu = st.radio(
        "Navigasi",
        [
            "Dashboard",
            "Chatbot AI",
            "Rekomendasi Wisata",
            "Data Destinasi",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        f"""
        <div class="profile-card">
            <div style="font-size:10px;color:#b8ded4;letter-spacing:1px;
                        text-transform:uppercase;margin-bottom:7px;">
                Pengembang Aplikasi
            </div>
            <div style="display:flex;gap:10px;align-items:center;">
                <div style="width:34px;height:34px;border-radius:50%;
                            background:#b9f3df;color:#07534d;display:flex;
                            align-items:center;justify-content:center;
                            font-weight:800;">R</div>
                <div>
                    <div class="profile-name">{NAMA_PENGEMBANG}</div>
                    <div class="profile-caption">Project SmartTour AI</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-size:10px;color:#b8ded4;margin-top:18px;'>"
        "Python · Streamlit · Gemini AI · TF-IDF</div>",
        unsafe_allow_html=True,
    )


# =========================================================
# 7. KOMPONEN TAMPILAN
# =========================================================
def tampilkan_hero(judul: str, deskripsi: str, label: str = "SMARTTOUR AI"):
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-eyebrow">{label}</div>
            <h1>{judul}</h1>
            <p>{deskripsi}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def tampilkan_kartu_destinasi(row: pd.Series):
    st.markdown(
        f"""
        <div class="soft-card">
            <div class="destination-title">📍 {row['nama_wisata']}</div>
            <div class="destination-meta">{row['lokasi']}</div>
            <span class="pill">{row['kategori']}</span>
            <div style="color:#496762;font-size:13px;line-height:1.6;
                        margin-top:10px;">{row['deskripsi']}</div>
            <div style="color:#66817d;font-size:11px;margin-top:10px;">
                <b>Aktivitas:</b> {row['aktivitas']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 8. HALAMAN DASHBOARD
# =========================================================
if menu == "Dashboard":
    tampilkan_hero(
        f"Selamat datang, Rangga! 👋",
        "Temukan inspirasi perjalanan, jelajahi destinasi, dan rencanakan "
        "pengalaman wisata di Provinsi Banten bersama asisten AI Anda.",
        "JELAJAHI KEINDAHAN BANTEN",
    )

    st.markdown(
        '<div class="section-kicker">Ringkasan data</div>'
        '<div class="section-heading">Wisata Banten dalam angka</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    metrics = [
        (col1, "🏝️", "Total Destinasi", len(df), "Destinasi pada dataset"),
        (col2, "📍", "Kabupaten/Kota", df["lokasi"].nunique(), "Wilayah terdata"),
        (col3, "✨", "Kategori Wisata", df["kategori"].nunique(), "Jenis wisata"),
    ]

    for column, icon, label, value, note in metrics:
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{icon} {label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown(
            '<div class="section-kicker">Eksplorasi</div>'
            '<div class="section-heading">Destinasi dalam dataset</div>',
            unsafe_allow_html=True,
        )
        kategori_count = (
            df["kategori"].value_counts().rename_axis("Kategori").reset_index(name="Jumlah")
        )
        fig = px.bar(
            kategori_count,
            x="Kategori",
            y="Jumlah",
            text="Jumlah",
            color="Kategori",
            color_discrete_sequence=[
                "#087e75", "#20a58e", "#70c9ad", "#b1e4d0", "#f0c987"
            ],
        )
        fig.update_layout(
            height=330,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=20, b=10),
            font=dict(family="DM Sans", color="#496762"),
            xaxis_title="",
            yaxis_title="Jumlah destinasi",
        )
        fig.update_traces(
            textposition="outside",
            marker_line_width=0,
            cliponaxis=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown(
            '<div class="section-kicker">Mulai perjalanan</div>'
            '<div class="section-heading">Apa yang ingin Anda cari?</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="soft-card">
                <div class="destination-title">💬 Tanya SmartTour AI</div>
                <div style="color:#66817d;font-size:13px;line-height:1.6;">
                    Tanyakan ide liburan, pilihan pantai, wisata alam,
                    sejarah, atau destinasi yang sesuai minat Anda.
                </div>
            </div>
            <div class="soft-card">
                <div class="destination-title">🧭 Temukan rekomendasi</div>
                <div style="color:#66817d;font-size:13px;line-height:1.6;">
                    Gunakan pencarian berbasis TF-IDF untuk menemukan
                    destinasi yang paling relevan dengan kata kunci Anda.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-kicker">Pilihan destinasi</div>'
        '<div class="section-heading">Jelajahi beberapa tempat wisata</div>',
        unsafe_allow_html=True,
    )
    preview = df.head(min(3, len(df)))
    cards = st.columns(3)
    for index, (_, row) in enumerate(preview.iterrows()):
        with cards[index]:
            tampilkan_kartu_destinasi(row)


# =========================================================
# 9. HALAMAN CHATBOT AI
# =========================================================
elif menu == "Chatbot AI":
    tampilkan_hero(
        "Teman perjalanan wisata Banten",
        "Ceritakan destinasi yang Anda cari. SmartTour AI akan membantu "
        "menemukan referensi wisata berdasarkan dataset Banten.",
        "SMARTTOUR AI CHAT",
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    top_col1, top_col2 = st.columns([5, 1])
    with top_col1:
        st.markdown(
            '<div class="section-heading">Percakapan Anda</div>',
            unsafe_allow_html=True,
        )
    with top_col2:
        if st.button("Hapus chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if not st.session_state.messages:
        st.info(
            "Halo Pengguna! 👋 Saya SmartTour AI. "
            "Silakan tanyakan destinasi wisata di Banten."
        )
        contoh = [
            "Rekomendasikan pantai di Banten",
            "Wisata alam apa yang ada di Pandeglang?",
            "Saya ingin wisata sejarah di Serang",
        ]
        cols = st.columns(3)
        for i, contoh_pertanyaan in enumerate(contoh):
            with cols[i]:
                st.caption(f"💡 {contoh_pertanyaan}")

    for pesan in st.session_state.messages:
        with st.chat_message(pesan["role"]):
            st.markdown(pesan["content"])

    pertanyaan = st.chat_input("Tulis pertanyaan tentang wisata Banten...")

    if pertanyaan:
        riwayat_sebelumnya = st.session_state.messages.copy()

        st.session_state.messages.append(
            {"role": "user", "content": pertanyaan}
        )

        with st.chat_message("user"):
            st.markdown(pertanyaan)

        hasil_rekomendasi = rekomendasi_wisata(pertanyaan, jumlah=5)
        konteks = format_konteks(hasil_rekomendasi)

        with st.chat_message("assistant"):
            with st.spinner("SmartTour AI sedang menyiapkan jawaban..."):
                jawaban = buat_jawaban_ai(
                    pertanyaan,
                    konteks,
                    riwayat_sebelumnya,
                )
            st.markdown(jawaban)

            if not hasil_rekomendasi.empty:
                with st.expander("Lihat destinasi yang dijadikan referensi"):
                    for _, row in hasil_rekomendasi.head(3).iterrows():
                        st.markdown(
                            f"**{row['nama_wisata']}** — {row['lokasi']} "
                            f"({row['kategori']})"
                        )

        st.session_state.messages.append(
            {"role": "assistant", "content": jawaban}
        )


# =========================================================
# 10. HALAMAN REKOMENDASI WISATA
# =========================================================
elif menu == "Rekomendasi Wisata":
    tampilkan_hero(
        "Temukan destinasi favorit Anda",
        "Pilih wilayah, kategori, dan minat perjalanan untuk mendapatkan "
        "rekomendasi berdasarkan kemiripan konten TF-IDF.",
        "REKOMENDASI CERDAS",
    )

    filter1, filter2 = st.columns(2)
    with filter1:
        pilihan_lokasi = st.selectbox(
            "📍 Kabupaten/Kota",
            ["Semua"] + sorted(df["lokasi"].astype(str).unique().tolist()),
        )
    with filter2:
        pilihan_kategori = st.selectbox(
            "🏷️ Kategori wisata",
            ["Semua"] + sorted(df["kategori"].astype(str).unique().tolist()),
        )

    minat = st.text_input(
        "🔎 Apa yang Anda sukai?",
        placeholder="Contoh: pantai, pemandangan laut, snorkeling, wisata alam",
    )

    hasil = df.copy()
    if pilihan_lokasi != "Semua":
        hasil = hasil[hasil["lokasi"] == pilihan_lokasi]
    if pilihan_kategori != "Semua":
        hasil = hasil[hasil["kategori"] == pilihan_kategori]

    if minat.strip():
        query_vector = vectorizer.transform([minat])
        matrix = vectorizer.transform(hasil["fitur"])
        scores = cosine_similarity(query_vector, matrix).flatten()
        hasil = hasil.copy()
        hasil["skor_kemiripan"] = scores
        hasil = hasil[hasil["skor_kemiripan"] > 0].sort_values(
            "skor_kemiripan", ascending=False
        )

    st.markdown(
        '<div class="section-heading">Hasil rekomendasi</div>',
        unsafe_allow_html=True,
    )

    if hasil.empty:
        st.warning(
            "Belum ditemukan destinasi yang sesuai. Coba kata kunci atau filter lain."
        )
    else:
        st.caption(f"{len(hasil)} destinasi ditemukan.")
        for _, row in hasil.iterrows():
            tampilkan_kartu_destinasi(row)


# =========================================================
# 11. HALAMAN DATA DESTINASI
# =========================================================
elif menu == "Data Destinasi":
    tampilkan_hero(
        "Database wisata Banten",
        "Jelajahi data destinasi yang menjadi sumber referensi "
        "rekomendasi dan jawaban Chatbot SmartTour AI.",
        "DATA DESTINASI",
    )

    search_col, count_col = st.columns([4, 1])
    with search_col:
        pencarian = st.text_input(
            "Cari destinasi",
            placeholder="Ketik nama destinasi, misalnya Pantai Anyer...",
        )
    with count_col:
        st.markdown(
            f"""
            <div class="metric-card" style="margin-top:28px;min-height:65px;">
                <div class="metric-label">Jumlah data</div>
                <div class="metric-value" style="font-size:23px;">{len(df)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    hasil = df.copy()
    if pencarian.strip():
        mask = (
            hasil["nama_wisata"].astype(str).str.contains(
                pencarian, case=False, na=False
            )
            | hasil["lokasi"].astype(str).str.contains(
                pencarian, case=False, na=False
            )
            | hasil["kategori"].astype(str).str.contains(
                pencarian, case=False, na=False
            )
        )
        hasil = hasil[mask]

    st.dataframe(
        hasil.drop(columns=["fitur"], errors="ignore"),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Menampilkan {len(hasil)} dari {len(df)} data destinasi.")


# =========================================================
# 12. FOOTER
# =========================================================
st.markdown(
    f"""
    <div class="footer">
        <b>SmartTour AI</b> · Banten Tourism Assistant<br>
        Dikembangkan oleh {NAMA_PENGEMBANG}<br>
        Python · Streamlit · Google Gemini API · TF-IDF
    </div>
    """,
    unsafe_allow_html=True,
)
