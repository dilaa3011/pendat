import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# Konfigurasi halaman
st.set_page_config(page_title="Pendat Personal", layout="wide")

# Muat dataset
df_data = pd.read_excel('balloons.xlsx')

# Judul dan deskripsi
st.title("Pendat Personal - Prediksi Balon")
st.write("Prediksi apakah balon akan terisi berdasarkan fitur yang dipilih.")

# Sidebar untuk input fitur
st.sidebar.header("Fitur Input")

# Kotak pilihan untuk setiap fitur dengan opsi default
selected_color = st.sidebar.selectbox('Pilih Warna:', [''] + list(df_data['color'].unique()), key='color')
selected_size = st.sidebar.selectbox('Pilih Ukuran:', [''] + list(df_data['size'].unique()), key='size')
selected_act = st.sidebar.selectbox('Pilih Akt:', [''] + list(df_data['act'].unique()), key='act')
selected_age = st.sidebar.selectbox('Pilih Umur:', [''] + list(df_data['age'].unique()), key='age')

# Periksa apakah semua input telah dipilih
if selected_color and selected_size and selected_act and selected_age:
    # Tangkap pilihan pengguna dalam new_data
    new_data = {
        'color': [selected_color],
        'size': [selected_size],
        'act': [selected_act],
        'age': [selected_age]
    }

    # Buat DataFrame dari new_data
    df_new = pd.DataFrame(new_data)

    # Encode kolom kategorikal menggunakan LabelEncoder
    label_encoders = {}
    categorical_columns = ['color', 'size', 'act', 'age']

    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        df_data[col] = label_encoders[col].fit_transform(df_data[col])  # Pastikan konsistensi encoding
        df_new[col] = label_encoders[col].fit_transform(df_new[col])  # Transform df_new

    # Load model
    with open('model.pkl', 'rb') as file:
        model_rfc = pickle.load(file)

    # Prediksi menggunakan model terlatih pada df_new
    new_predictions = model_rfc.predict(df_new)

    # Peta nilai yang diprediksi ke 'T' atau 'F'
    df_new['Predicted'] = new_predictions

    # Tampilkan hasil prediksi
    st.subheader("Hasil Prediksi")

    st.write(df_new[['color', 'size', 'act', 'age', 'Predicted']])

    # Evaluasi model pada set pengujian (opsional jika Anda ingin menampilkan akurasi)
    X = df_data[['color', 'size', 'act', 'age']]
    y = df_data['inflated']
    y_pred = model_rfc.predict(X)
    accuracy = accuracy_score(y, y_pred)
    st.write(f"Akurasi pada seluruh dataset: {accuracy * 100:.2f}%")
    st.write('note :')
    st.write('1. Jika tabel Predicted menyatakan F maka balon akan mengembang')
    st.write('2. jika tabel Predicted menyatakan T maka balon tidak mengembang')
    st.link_button('My Github','https://github.com/dilaa3011/pendat')
else:
    st.sidebar.write("Silakan pilih nilai untuk semua fitur.")
    st.link_button('My Github','https://github.com/dilaa3011/pendat')
