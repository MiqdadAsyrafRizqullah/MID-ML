import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

@st.cache_resource
def load_and_train_model():
    # 1. Memuat Dataset Asli
    df_kaggle = pd.read_csv('StressLevelDataset.csv')
    df_gform = pd.read_csv('DatasetStressMiqdad - Form Responses 1.csv')
    
    # 2. Preprocessing Data Google Form persis seperti di Notebook
    df_gform = df_gform.drop('Timestamp', axis=1, errors='ignore')
    
    # Encoding
    if 'mental_health_history' in df_gform.columns and df_gform['mental_health_history'].dtype == 'object':
        df_gform['mental_health_history'] = df_gform['mental_health_history'].map({'Ya': 1, 'Tidak': 0})
    
    if 'stress_level' in df_gform.columns and df_gform['stress_level'].dtype == 'object':
        df_gform['stress_level'] = df_gform['stress_level'].astype(str).str.extract(r'\((\d)\)')[0].astype(float).astype(int)
        
    df_gform.drop_duplicates(inplace=True)
    
    # Penyesuaian nama kolom GForm agar cocok dengan Kaggle
    df_gform.rename(columns={'living_conditions]': 'living_conditions'}, inplace=True)
    
    # Penyesuaian Skala Data Google Form (1-5 ke skala Kaggle)
    # Catatan: Kaggle memiliki rentang berbeda-beda (0-21, 0-3, 0-5, dll)
    mapping_config = {
        'anxiety_level': 21,
        'depression': 27,
        'self_esteem': 30,
        'mental_health_history': 1,
        'academic_performance': 5,
        'study_load': 5,
        'teacher_student_relationship': 5,
        'future_career_concerns': 5,
        'social_support': 3,
        'peer_pressure': 5,
        'bullying': 5,
        'headache': 5,
        'sleep_quality': 5,
        'breathing_problem': 5,
        'blood_pressure': 3,
        'basic_needs': 5,
        'noise_level': 5,
        'living_conditions': 5,
        'safety': 5,
        'extracurricular_activities': 5
    }
    
    # Hanya lakukan scaling pada kolom yang ada di df_gform dan bernilai numerik
    # mental_health_history TIDAK boleh diskalakan karena sudah 0/1
    scalable_cols = {k: v for k, v in mapping_config.items() if k != 'mental_health_history'}
    
    for col, m_val in scalable_cols.items():
        if col in df_gform.columns:
            # Pastikan kolom adalah numerik sebelum dihitung
            df_gform[col] = pd.to_numeric(df_gform[col], errors='coerce').fillna(1)
            # Formula: GForm 1-5 -> Kaggle 0-m_val
            df_gform[col] = ((df_gform[col] - 1) / 4) * m_val
    
    # Pastikan urutan dan nama kolom GForm mutlak sama dengan Kaggle
    df_gform = df_gform[df_kaggle.columns]
            
    # Oversampling Google Form (Domain Adaptation)
    df_gform_train = df_gform.sample(frac=0.7, random_state=42)
    df_oversampled = pd.concat([df_gform_train]*30, ignore_index=True)
    
    # Penggabungan Final
    df_combined = pd.concat([df_kaggle, df_oversampled], ignore_index=True)
    df_fe = df_combined.copy()
    
    # 3. Feature Engineering (WAJIB SAMA ANTARA TRAIN & PREDICT)
    df_fe['psychological_burden'] = df_fe['anxiety_level'] + df_fe['depression']
    df_fe['academic_pressure'] = df_fe['study_load'] + (5 - df_fe['academic_performance'])
    df_fe['social_stress'] = df_fe['peer_pressure'] + (3 - df_fe['social_support'])
    df_fe['physical_symptom'] = df_fe['headache'] + df_fe['breathing_problem'] + df_fe['blood_pressure']
    df_fe['environment_risk'] = df_fe['noise_level'] + (5 - df_fe['living_conditions']) + (5 - df_fe['safety'])
    df_fe['wellbeing_index'] = df_fe['sleep_quality'] + df_fe['basic_needs'] + df_fe['self_esteem']
    df_fe['stress_interaction'] = df_fe['anxiety_level'] * df_fe['depression']
    
    X = df_fe.drop('stress_level', axis=1).apply(pd.to_numeric, errors='coerce').fillna(0)
    y = pd.to_numeric(df_fe['stress_level'], errors='coerce').fillna(0)
    
    # 4. Melatih Model Terbaik (Tanpa Scaling Eksternal, sesuai prinsip BAB 7)
    model = RandomForestRegressor(random_state=42, n_estimators=150, max_depth=5)
    model.fit(X, y)
    
    return model, X.columns.tolist()

# ==================== CONFIG UI STREAMLIT ====================
st.set_page_config(page_title="Deteksi Stres Mahasiswa", page_icon="🧠", layout="wide")

st.title("🧠 Aplikasi Deteksi Tingkat Stres & Risiko Depresi Mahasiswa")
st.markdown("Aplikasi berbasis **Machine Learning (Model Evaluasi: Random Forest Regressor)** dari penyusun Miqdad Asyraf Rizqullah untuk memprediksi tingkat stres berdasarkan **20 Fakor Input** psikologis, akademik, gaya hidup, dan sosial.")

with st.spinner("Memuat dan mempersiapkan model Machine Learning... (Hanya butuh 1-2 detik)"):
    model, feature_cols = load_and_train_model()

st.header("📝 FORM INPUT MAHASISWA")
st.markdown("Silakan atur skala pada parameter di bawah ini untuk memprediksi tingkat stres Anda/responden:")

# ==================== INPUT FIELDS (MAIN AREA) ====================
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("1. Dimensi Psikologis")
    anxiety_level = st.slider("Kecemasan (Anxiety Level)", 0, 21, 10, help="0: Sangat Santai, 21: Sangat Cemas")
    depression = st.slider("Depresi (Depression)", 0, 27, 10, help="0: Sangat Bahagia, 27: Menunjukkan Gejala Depresi Kuat")
    self_esteem = st.slider("Harga Diri (Self Esteem)", 0, 30, 15, help="Tingkat kepercayaan diri / menghargai diri yang sehat")
    mental_health_history = st.selectbox("Riwayat Kesehatan Mental", [0, 1], format_func=lambda x: "Ya (Ada Riwayat)" if x==1 else "Tidak (Sehat)")

    st.subheader("3. Gaya Hidup & Fisik")
    sleep_quality = st.slider("Kualitas Tidur", 0, 5, 3, help="0: Insomnia/Buruk Sekali, 5: Sangat Nyenyak")
    headache = st.slider("Frekuensi Sakit Kepala", 0, 5, 1)
    blood_pressure = st.slider("Rentan Tekanan Darah (1=Rendah, 2=Normal, 3=Tinggi)", 1, 3, 2)
    breathing_problem = st.slider("Masalah Pernapasan (Sesak saat panik)", 0, 5, 0)
    basic_needs = st.slider("Kecukupan Kebutuhan Dasar (Ekonomi/Pangan)", 0, 5, 4)
    extracurricular = st.slider("Aktif Ekstrakurikuler / UKM", 0, 5, 2)

with col_right:
    st.subheader("2. Dimensi Akademik")
    study_load = st.slider("Beban Belajar (Study Load)", 0, 5, 2)
    academic_performance = st.slider("Performa Akademik (Nilai/IPK)", 0, 5, 3)
    future_career_concerns = st.slider("Kekhawatiran Karir Masa Depan", 0, 5, 3)
    teacher_student_rel = st.slider("Hubungan dengan Dosen", 0, 5, 4)

    st.subheader("4. Sosial & Lingkungan")
    social_support = st.slider("Dukungan Sosial/Keluarga", 0, 3, 2)
    peer_pressure = st.slider("Tekanan Teman Sebaya (Peer Pressure)", 0, 5, 1)
    bullying = st.slider("Sering Mengalami Bullying?", 0, 5, 0)
    living_conditions = st.slider("Kenyamanan Tempat Tinggal", 0, 5, 4)
    safety = st.slider("Rasa Aman di Lingkungan", 0, 5, 4)
    noise_level = st.slider("Tingkat Kebisingan Radius Tinggal", 0, 5, 1)

st.markdown("---")
predict_btn = st.button("🚀 PREDIKSI STRES SEKARANG!", use_container_width=True)

# Layout Image
if not predict_btn:
    st.info("👈 **Panduan Penggunaan:** Atur kombinasi data di atas lalu klik tombol **PREDIKSI STRES SEKARANG!**")
    st.image("https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?q=80&w=1470&auto=format&fit=crop", caption="Artificial Intelligence System - Prediksi Stres Mahasiswa", use_container_width=True)

# ==================== PREDICTION LOGIC ====================
if predict_btn:
    # 1. Tampung dictionary data
    input_data = {
        'anxiety_level': anxiety_level,
        'self_esteem': self_esteem,
        'mental_health_history': mental_health_history,
        'depression': depression,
        'headache': headache,
        'blood_pressure': blood_pressure,
        'sleep_quality': sleep_quality,
        'breathing_problem': breathing_problem,
        'noise_level': noise_level,
        'living_conditions': living_conditions,
        'safety': safety,
        'basic_needs': basic_needs,
        'academic_performance': academic_performance,
        'study_load': study_load,
        'teacher_student_relationship': teacher_student_rel,
        'future_career_concerns': future_career_concerns,
        'social_support': social_support,
        'peer_pressure': peer_pressure,
        'extracurricular_activities': extracurricular,
        'bullying': bullying,
    }
    
    df_input = pd.DataFrame([input_data])
    
    # 2. Evaluasi menggunakan Feature Engineering (Identik dengan Notebook)
    df_input['psychological_burden'] = df_input['anxiety_level'] + df_input['depression']
    df_input['academic_pressure'] = df_input['study_load'] + (5 - df_input['academic_performance'])
    df_input['social_stress'] = df_input['peer_pressure'] + (3 - df_input['social_support'])
    df_input['physical_symptom'] = df_input['headache'] + df_input['breathing_problem'] + df_input['blood_pressure']
    df_input['environment_risk'] = df_input['noise_level'] + (5 - df_input['living_conditions']) + (5 - df_input['safety'])
    df_input['wellbeing_index'] = df_input['sleep_quality'] + df_input['basic_needs'] + df_input['self_esteem']
    df_input['stress_interaction'] = df_input['anxiety_level'] * df_input['depression']
    
    # Terapkan penyelarasan urutan kolom mutlak agar tidak error
    df_input = df_input[feature_cols]
    
    # 3. Eksekusi Prediksi
    pred_raw = model.predict(df_input)[0]
    
    # Terapkan metode Clamped-Regression Notebook (Membulatkan dan Limitasi Max 2 dan Min 0)
    pred_rounded = int(np.clip(np.round(pred_raw), 0, 2))
    
    st.markdown("---")
    st.header("🎯 KESIMPULAN / HASIL PREDIKSI")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Skor Prediksi Mentah (Desimal Mesin)", value=f"{pred_raw:.3f}", delta="Akurasi Model Eksternal ~82%", delta_color="normal")
        st.caption("Skala mentah menunjukkan pergerakan seberapa dekat dari perpindahan jenjang tipe stres.")
        
    with col2:
        if pred_rounded == 0:
            st.success("Tingkat Stres: **RENDAH (0)** 😊")
            st.write("Kesehatan mental mahasiswa berada dalam kondisi yang sangat stabil. Tingkat dukungan sosial, beban psikologis yang rendah, dan gaya hidup sehat telah membentuk sistem imun psikis yang baik. Pertahankan!")
        elif pred_rounded == 1:
            st.warning("Tingkat Stres: **SEDANG (1)** 😐")
            st.write("Terdapat indikasi kemunculan tekanan baik secara akademik maupun sosial. Jangan sepelekan, atur ulang porsi istirahat, manajemen waktu, serta mulailah menceritakan masalah lingkungan kepada orang terdekat.")
        else:
            st.error("Tingkat Stres: **TINGGI (2)** 🚨")
            st.write("RISIKO DEPRESI KRITIS! Berdasarkan kombinasi kecemasan, kelelahan mental, masalah akademik, maupun minimnya dukungan moral, mahasiswa dinyatakan pada level tekanan emosional serius. **Segera perbaiki pola tidur dan cari layanan / konseling bantuan Universitas.**")

    st.markdown("---")
    
    # Visualisasi Radar tambahan untuk Insight User (Opsional tapi Wow!)
    st.subheader("📊 Analisis Komponen Stres Berdasarkan Input")
    st.write("Mesin menyoroti bobot komponen dari respons Anda yang bisa jadi penggerak utama:")
    
    comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
    comp_col1.metric("Beban Psikologis", df_input['psychological_burden'].values[0])
    comp_col2.metric("Tekanan Akademik", df_input['academic_pressure'].values[0])
    comp_col3.metric("Tekanan Sosial", df_input['social_stress'].values[0])
    comp_col4.metric("Kesehatan Fisik & Tidur", df_input['physical_symptom'].values[0])

