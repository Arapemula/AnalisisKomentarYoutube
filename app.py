import os
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Baris ini penting untuk deployment agar Matplotlib berjalan tanpa GUI
import matplotlib.pyplot as plt
import itertools
import time
from flask import Flask, render_template, request, url_for, redirect

from youtube_comment_downloader import YoutubeCommentDownloader
from transformers import pipeline

# --- KONFIGURASI APLIKASI FLASK ---
app = Flask(__name__)
# Menonaktifkan caching agar gambar plot selalu yang terbaru
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# --- DEFINISI ID MODEL DARI HUGGING FACE HUB ---
# GANTI DENGAN ID MODEL ANDA SENDIRI YANG SUDAH DIUNGGAH KE HUGGING FACE HUB!
SENTIMENT_MODEL_ID = "username-anda/sentimen-komentar-youtube-indo" 
EMOTION_MODEL_ID = "MarfinF/marfin_emotion"

print(f"INFO: Memuat model sentimen dari Hub: {SENTIMENT_MODEL_ID}")
print(f"INFO: Memuat model emosi dari Hub: {EMOTION_MODEL_ID}")

# --- PEMUATAN MODEL (DILAKUKAN SEKALI SAAT STARTUP) ---
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_ID)
    emotion_analyzer = pipeline("text-classification", model=EMOTION_MODEL_ID)
    print("INFO: Model sentimen dan emosi berhasil dimuat.")
except Exception as e:
    print(f"ERROR: Gagal memuat model. Error: {e}")
    sentiment_analyzer = None
    emotion_analyzer = None

# --- FUNGSI-FUNGSI LOGIKA APLIKASI ---

def scrape_youtube_comments(video_url, limit=200):
    """Mengambil komentar dari URL YouTube dengan batas yang ditentukan."""
    print(f"INFO: Mengambil komentar dari YouTube (Batas: {limit}) untuk URL: {video_url}")
    downloader = YoutubeCommentDownloader()
    try:
        comments_generator = downloader.get_comments_from_url(video_url, sort_by=1)
        comment_data = [{'Comment': str(comment['text'])} for comment in itertools.islice(comments_generator, limit)]
        
        if not comment_data:
            print("WARNING: Tidak ada komentar yang berhasil diambil.")
            return None, "Tidak ada komentar yang ditemukan pada video ini."
        
        df = pd.DataFrame(comment_data).dropna(subset=['Comment'])
        print(f"INFO: {len(df)} komentar berhasil diambil.")
        return df, None
    except Exception as e:
        error_message = str(e)
        print(f"ERROR saat mengambil komentar: {error_message}")
        if "unavailable" in error_message.lower() or "not found" in error_message.lower():
            return None, "Video tidak ditemukan atau tidak tersedia. Pastikan URL benar."
        return None, "Terjadi kesalahan saat mencoba mengambil komentar dari YouTube."

def run_full_analysis(comments_df):
    """Menjalankan analisis berlapis dan mengembalikan DataFrame serta nama file gambar plot."""
    if comments_df is None or sentiment_analyzer is None or emotion_analyzer is None:
        return None, None

    print("INFO: Memulai analisis berlapis...")

    # LAPISAN 1: ANALISIS SENTIMEN
    def get_sentiment(text):
        if not isinstance(text, str) or len(text.strip()) == 0: return 'neutral'
        try:
            return sentiment_analyzer(text[:512])[0]['label'].lower()
        except Exception:
            return 'neutral'
    comments_df['sentimen'] = comments_df['Comment'].apply(get_sentiment)
    print("INFO: Analisis sentimen selesai.")

    # LAPISAN 2: ANALISIS EMOSI
    def get_emotion(row):
        if row['sentimen'] in ['positive', 'negative']:
            try:
                return emotion_analyzer(row['Comment'][:512])[0]['label']
            except Exception:
                return 'tidak diketahui'
        else:
            return 'netral'
    comments_df['emosi'] = comments_df.apply(get_emotion, axis=1)
    print("INFO: Analisis emosi selesai.")

    # --- VISUALISASI ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Sentimen
    sentiment_counts = comments_df['sentimen'].value_counts()
    colors_sentiment = {'positive': 'lightgreen', 'negative': 'salmon', 'neutral': 'lightblue'}
    pie_colors_sent = [colors_sentiment.get(key, 'gray') for key in sentiment_counts.index]
    axes[0].pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=pie_colors_sent)
    axes[0].set_title('Distribusi Sentimen')

    # Plot 2: Emosi (Non-Netral)
    emotion_counts = comments_df[comments_df['emosi'] != 'netral']['emosi'].value_counts()
    if not emotion_counts.empty:
        axes[1].pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=140)
        axes[1].set_title('Distribusi Emosi (Non-Netral)')
    else:
        axes[1].text(0.5, 0.5, 'Tidak ada data emosi\n(semua sentimen netral)', ha='center', va='center')
        axes[1].set_title('Distribusi Emosi (Non-Netral)')
        axes[1].axis('off')

    plt.suptitle('Hasil Analisis Komentar YouTube', fontsize=16)
    
    # Simpan plot sebagai file gambar di folder static
    image_name = f'hasil_analisis_{int(time.time())}.png'
    image_path = os.path.join('static', 'images', image_name)
    plt.savefig(image_path)
    plt.close(fig) # Tutup plot agar tidak memakan memori
    print(f"INFO: Visualisasi disimpan di '{image_path}'")

    return comments_df, image_name

# --- ROUTE / HALAMAN WEB ---

@app.route('/')
def home():
    """Menampilkan halaman utama (index.html)."""
    return render_template('index.html')

@app.route('/analisis', methods=['POST'])
def analisis():
    """Menerima request, menjalankan analisis, dan menampilkan hasil."""
    youtube_url = request.form['youtube_url']
    if not youtube_url.strip():
        return render_template('index.html', error="URL tidak boleh kosong.")

    # 1. Scrape komentar dan tangani error
    comments_df, error = scrape_youtube_comments(youtube_url)
    if error:
        return render_template('index.html', error=error, prev_url=youtube_url)

    # Tampilkan halaman loading jika ada banyak komentar (opsional, untuk UX yang lebih baik)
    # Di sini kita langsung proses
    
    # 2. Jalankan analisis lengkap
    hasil_df, image_filename = run_full_analysis(comments_df)
    if hasil_df is None:
        return render_template('index.html', error="Terjadi kesalahan internal saat menganalisis komentar.", prev_url=youtube_url)
        
    # 3. Buat path URL untuk gambar
    image_url = url_for('static', filename=f'images/{image_filename}')

    # 4. Kirim data ke template hasil.html
    return render_template(
        'hasil.html', 
        youtube_url=youtube_url, 
        image_file=image_url,
        comments=hasil_df.to_dict(orient='records'),
        total_comments=len(hasil_df)
    )

if __name__ == '__main__':
    # Pastikan folder untuk gambar ada
    if not os.path.exists(os.path.join('static', 'images')):
        os.makedirs(os.path.join('static', 'images'))
    app.run(debug=True)