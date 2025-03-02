import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

pip==23.0

import gdown

url = "https://drive.google.com/file/d/1vYT1Y5aal597A-bVUw3uuCKw2mGxBa8-/view?usp=drive_link"
output = "model.h5"
gdown.download(url, output, quiet=False)


# Fungsi untuk memuat model dan melakukan prediksi
def load_and_predict(image_path):
    IMAGE_SIZE = (224, 224)
    
    # Membaca gambar
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMAGE_SIZE)  # Resize gambar sesuai ukuran input model
    img = img.astype('float32') / 255.0  # Normalisasi
    
    # Menambah dimensi batch
    img = np.expand_dims(img, axis=0)  # Menambahkan dimensi batch (1,)
    
    # Memuat model yang sudah dilatih
    model = load_model('model_penyakit_padi.h5')
    
    # Melakukan prediksi
    prediction = model.predict(img)
    
    # Menampilkan kelas dengan probabilitas tertinggi
    predicted_class = np.argmax(prediction)
    
    # Definisikan label untuk kelas
    classes = ['Tungro', 'Blast', 'Blight']
    
    # Kembalikan kelas yang diprediksi dan probabilitasnya
    return classes[predicted_class], prediction[0][predicted_class]

st.markdown("""
    <style>
    .header {
        text-align: center;
        font-size: 35px;
        font-weight: bold;
        margin-top: 50px;
        color: white;  
        background-color: #2c7839; 
        text-shadow: 2px 2px 5px rgba(0,0,0,0.7);
        padding: 20px;
    }
    .content {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 30px;
        border-radius: 10px;
        margin-top: 30px;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: #777;
        margin-top: 50px;
    }
    .image-container {
        text-align: center;
    }
    .example-img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
        border-radius: 10px;
        box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.1);
    }
    .predictions {
        margin-top: 20px;
        text-align: center;
    }
    .prediction-text {
        font-size: 20px;
        color: white;  
        font-weight: bold;
    }
    .recommendation {
        background-color: rgba(255, 255, 255, 0.7);
        padding: 20px;
        border-radius: 10px;
        font-size: 16px;
        margin-top: 20px;
    }
    .emoji {
        font-size: 50px;
        text-align: center;
    }
    .button {
        background-color: #2c7839;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 5px;
        text-decoration: none;
        display: inline-block;
        margin: 10px 0;
    }
    .button:hover {
        background-color: #1d5d29;
    }
    </style>
    <div class="header">
        🌾(LPDP) Layanan Pendeteksi Daun Padi
    </div>
""", unsafe_allow_html=True)

# Rekomendasi Pengobatan untuk setiap penyakit
def get_treatment_recommendation(disease):
    recommendations = {
        "Tungro": (
            "Tungro adalah penyakit yang disebabkan oleh virus yang menyerang tanaman padi. "
            "Berikut adalah beberapa rekomendasi pengobatan untuk penyakit Tungro:\n\n"
            "1. Gunakan varietas padi yang tahan terhadap penyakit Tungro.\n"
            "2. Terapkan rotasi tanaman untuk mengurangi jumlah vektor yang membawa virus.\n"
            "3. Gunakan insektisida untuk mengendalikan serangga yang menjadi vektor penyakit.\n"
            "4. Hapus tanaman yang terinfeksi untuk mencegah penyebaran penyakit."

        ),
        "Blast": (
            "Blast (Pyricularia oryzae) adalah penyakit jamur yang sering menyerang tanaman padi. "
            "Berikut adalah beberapa rekomendasi pengobatan untuk penyakit Blast:\n\n"
            "1. Gunakan varietas padi yang tahan terhadap Blast.\n"
            "2. Aplikasikan fungisida sesuai petunjuk untuk mengendalikan jamur.\n"
            "3. Jaga kebersihan lahan dengan menghilangkan sisa tanaman yang terinfeksi.\n"
            "4. Pastikan pengairan yang baik untuk mencegah kelembaban berlebih yang mendukung perkembangan jamur."
        ),
        "Blight": (
            "Blight (Bacterial Blight) adalah penyakit yang disebabkan oleh bakteri dan dapat mengakibatkan kerusakan serius pada tanaman padi. "
            "Berikut adalah beberapa rekomendasi pengobatan untuk penyakit Blight:\n\n"
            "1. Gunakan varietas padi yang tahan terhadap Blight.\n"
            "2. Terapkan rotasi tanaman untuk mengurangi patogen di tanah.\n"
            "3. Gunakan antibiotik atau agen pengendali biologi untuk mengurangi infeksi bakteri.\n"
            "4. Hapus tanaman yang terinfeksi dan sanitasi alat pertanian untuk mencegah penyebaran penyakit."
        )
    }
    return recommendations.get(disease, "Rekomendasi pengobatan tidak tersedia untuk penyakit ini.")

# Deskripsi aplikasi
st.write("""
    Aplikasi ini digunakan untuk mendeteksi penyakit pada daun padi (Tungro, Blast, Blight).
    Silakan upload gambar daun padi untuk mendapatkan prediksi penyakit.
""")
st.write("""
         untuk mendapatkan hasil prediksi yang maksimal, pastikan gambar yang diupload dapat terlihat dengan jelas,
         dan gunakan latar belakang putih jika diperlukan
         """)

st.markdown("### Contoh Gambar Ideal untuk Deteksi Penyakit Padi")
col1, col2, col3 = st.columns(3)  
example_images = [
    "tungro.jpg",  # Ganti dengan path gambar Anda
    "blast.jpg",  # Ganti dengan path gambar Anda
    "blight.jpg"  # Ganti dengan path gambar Anda
]

with col1:
    example_img1 = Image.open(example_images[0])
    st.image(example_img1, caption="Contoh Gambar Tungro", width=150)

with col2:
    example_img2 = Image.open(example_images[1])
    st.image(example_img2, caption="Contoh Gambar Blast", width=150)

with col3:
    example_img3 = Image.open(example_images[2])
    st.image(example_img3, caption="Contoh Gambar Blight", width=150)

# Sidebar untuk pengaturan
st.sidebar.header("Pengaturan Tampilan")
max_width = st.sidebar.slider("Pilih Lebar Gambar", min_value=200, max_value=600, value=300)

# Membuat area upload gambar
uploaded_file = st.file_uploader("Upload Gambar Daun Padi", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menampilkan gambar yang di-upload dengan ukuran yang dikendalikan
    st.image(uploaded_file, caption="Gambar yang di-upload", width=max_width)  # Menggunakan parameter width
    
    # Mengkonversi gambar ke format yang bisa dibaca oleh OpenCV
    img = Image.open(uploaded_file)
    img_path = "uploaded_image.jpg"
    img.save(img_path)  # Menyimpan gambar sementara untuk diproses
    
    # Menjalankan prediksi
    predicted_label, predicted_prob = load_and_predict(img_path)
    
    # Menampilkan hasil prediksi dengan tampilan yang lebih menarik
    st.markdown(f"### **Penyakit yang terdeteksi**: {predicted_label}")
    st.markdown(f"### **Probabilitas**: {predicted_prob * 100:.2f}%")
    
    # Menampilkan rekomendasi pengobatan untuk penyakit yang terdeteksi
    treatment_recommendation = get_treatment_recommendation(predicted_label)
    st.write(treatment_recommendation)


# Menambahkan footer atau pesan penutup
st.markdown("""
    <br><br>
    <footer style="text-align: center; font-size: 14px;">
        Dibuat oleh [Didit-Siti-Chrisma] - Rice Crop Disease Detection
            <p>Untuk informasi lebih lanjut atau konsultasi, silakan hubungi kami:</p>
        <a href="https://wa.me/+6281299336645" class="button" ; font-weight: bold;" target="_blank">Chat via WhatsApp</a>
        <br>
        <a href="https://www.pupuk-indonesia.com/product/petani" class="button" ; font-weight: bold;" target="_blank">Kunjungi Website Kami</a>
    </footer>
""", unsafe_allow_html=True)
