from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_DIR = "./model_hasil_tuning" 


NAMA_MODEL_DI_HUB = "agus1111/sentimen-komentar-youtube-indo"

print(f"Memuat model dari direktori lokal: {MODEL_DIR}")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

print(f"Model dan tokenizer berhasil dimuat. Mengunggah ke Hub sebagai '{NAMA_MODEL_DI_HUB}'...")


model.push_to_hub(NAMA_MODEL_DI_HUB)
tokenizer.push_to_hub(NAMA_MODEL_DI_HUB)

print("✅ Selesai! Model Anda sekarang sudah ada di Hugging Face Hub.")