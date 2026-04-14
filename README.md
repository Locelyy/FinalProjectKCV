# Klasifikasi Penyakit Payudara (Breast Disease Classification)

Project ini dibuat untuk mengklasifikasikan histopatologi kanker payudara menggunakan Deep Learning dengan **ResNet50, EfficientNet-B5, densenet121** dengan metode ensemble.

## Sumber Dataset
**BreaKHis - Breast Cancer Histopathological Database**  
Dataset dapat diunduh pada tautan berikut: [Mendeley Data - BreaKHis](https://data.mendeley.com/datasets/jxwvdwhpc2/1)

## Persiapan Lingkungan (*Environment Setup*)
1. Letakkan dataset yang telah diekstrak ke dalam folder `data\raw\` berdasarkan skala perbesaran (40X, 100X, 200X, 400X) dengan mematuhi struktur di bawah ini:
```text
C:.
├───data
│   ├───raw
│   │   ├───40X
│   │   │   ├───adenosis
│   │   │   ├───ductal_carcinoma
│   │   │   ├───fibroadenoma
│   │   │   ├───lobular_carcinoma
│   │   │   ├───mucinous_carcinoma
│   │   │   ├───papillary_carcinoma
│   │   │   ├───phyllodes_tumor
│   │   │   └───tubular_adenoma
│   │   ├───100X
│   │   │   └─── ...
│   │   ├───200X
│   │   │   └─── ...
│   │   └───400X
│   │       └─── ...
```

2. Buat *virtual environment*:
```bash
python -m venv venv
```

3. Aktifkan *virtual environment*:
- **Windows**: `venv\Scripts\activate`
- **Linux/Mac**: `source venv/bin/activate`

4. Instal paket dependensi pendukung:
```bash
pip install -r requirements.txt
```

## Persiapan Data (*Data Preparation*)
Jalankan skrip berikut secara beurutan guna mengekstrak metadata dari hirarki folder dataset dan kemudian membaginya sesuai dengan proporsi pelatihan (*training*) dan pengujian (*testing*):
1. Menggabungkan informasi dan membuat *metadata* ke dalam file CSV:
```bash
python src/make_metadata_all.py
```
2. Membagi dataset (*Train/Test Split*):
```bash
python src/split_data_all.py
```

## Pelatihan dan Evaluasi Model (*Model Training & Evaluation*)
Untuk menjalankan proses training pada tiap model (ResNet50, EfficientNet-B5, densenet121) dapat dilakukan dengan cara:
1. Menjalankan proses pelatihan:
```bash
python src/train_[nama model].py

Example:
python src/train_efficientnet_b5.py
```
2. Mengevaluasi akurasi performa model menggunakan data pengujian (*testing set*):
```bash
python src/evaluate_[nama model].py
```

## Prediksi Manual dan Aplikasi Web (*Inference & Web App*)
Anda dapat menjalankan prediksi dengan menggunakan user interface yang dibuat di **Streamlit**.

1. Melakukan prediksi manual untuk gambar tertentu melalui terminal:
```bash
python src/infer_[nama model].py
```

2. **Menjalankan Aplikasi Web (Sangat Direkomendasikan):**
Aplikasi web ini memuat tiga model yaitu **ResNet50, EfficientNet-B5, densenet121** yang dijalankan dengan *Heatmap / Grad-CAM* untuk mendeteksi kanker pada gambar yang diunggah ke website.
```bash
streamlit run app/app.py
```
