# 🧠 Prediksi Tingkat Stres & Risiko Depresi Mahasiswa

Project Machine Learning ini dibuat untuk mendeteksi awal risiko kesehatan mental (stres dan depresi) tingkat mahasiswa menggunakan model **Random Forest Regressor** (dengan R2 Score >90%). Model ini dilatih menggunakan kombinasi dataset sekunder dari Kaggle dan data primer langsung dari survei independen (Google Forms) yang menerapkan tahapan *Feature Engineering* kompleks.

Proyek ini tidak hanya berisi hasil riset komprehensif, tetapi juga menyertakan **Aplikasi Berbasis Web (Streamlit)

---

## 🚀 Fitur Utama
1. **Analisis Data & Prediksi**: Memprediksi tingkat stres pada skala 0 (Rendah), 1 (Sedang), dan 2 (Tinggi).
2. **20 Indikator Masukan**: Menilainya melalui faktor *Psikologis*, *Akademik*, *Gaya Hidup*, dan *Lingkungan/Sosial*.
3. **Web Aplikasi Streamlit**: Interface/UI yang mudah digunakan oleh kalangan awam.

---

## 💻 Persiapan dan Instalasi (Bisa di Windows, Mac, maupun Linux)

Bagi pengguna (teman, kolega, dosen) yang ingin menjalankan proyek dan aplikasi ini secara instan di komputer masing-masing; persiapkan **Python (versi 3.8 ke atas)** lalu ikuti 3 langkah mudah ini:

### 1. Download/Clone Project Ini
Pastikan semua file berikut berada secara lengkap di dalam satu folder yang sama di laptop Anda:
- `app.py` (File antarmuka/Streamlit asli)
- `E1E124041_MIQDAD_PEJUANG_MACHINE LEARNING_FINAL_BESOK1945.ipynb`
- `DatasetStressMiqdad - Form Responses 1.csv`
- `StressLevelDataset.csv`
- `requirements.txt`

### 2. Instal Library yang Dibutuhkan
Buka terminal (Mac/Linux) atau Command Prompt / PowerShell (Windows), dan navigasikan/arahkah ke dalam folder project (`cd path/ke/folder/project`). Tuliskan perintah sakti ini:

```bash
pip install -r requirements.txt
```
*Atau, apabila Anda tidak memiliki file `requirements.txt` / gagal, jalankan perintah manual ini:*
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit fpdf2 PyPDF2
```

### 3. Cara Menjalankan Aplikasi Web (Mencoba Sendiri Prediksinya)
Sesudah proses instalasi sukses, jalankan kode ini di terminal yang masih terbuka pada folder yang sama:
```bash
streamlit run app.py
```
> **Browser Otomatis Terbuka**: Anda akan diarahkan ke link `http://localhost:8501`. Dan aplikasi web AI Prediksi Stres SIAP digunakan di PC tersebut! 

---

---

**Hak Cipta Riset & Pemodelan Machine Learning**: 
Miqdad Asyraf Rizqullah (E1E124041) - *Pejuang Machine Learning*
