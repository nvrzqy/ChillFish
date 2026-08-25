# ChillFish AI MVP

ChillFish adalah aplikasi AI decision-support untuk membantu membaca risiko kualitas lot ikan lemuru (`Sardinella lemuru`) pada konteks cold-chain dan logistik PPN Pengambengan. Aplikasi ini menerima input lot dari dataset atau input manual, lalu menampilkan prediksi kondisi lot, risk score, rekomendasi operasional, probabilitas model, key signals, dan visual anomaly jika pengguna mengunggah foto.

Output ChillFish bersifat pendukung keputusan. Sistem ini bukan alat sertifikasi keamanan pangan, bukan pengganti pengujian histamin, bukan pengganti standar SNI, dan bukan keputusan ekspor final.

## Fitur Utama

- Dataset Mode untuk memilih/search lot dari dataset lokal.
- Manual Mode untuk memasukkan data lot baru.
- Optional upload foto untuk visual anomaly screening.
- Prediksi `Condition`, `Risk Score`, dan `Recommendation`.
- Probabilitas untuk action dan condition.
- Toggle bahasa English/Indonesia.
- Export prediksi dataset ke CSV.
- Berjalan lokal menggunakan Docker Compose.

## Arsitektur Sistem

```text
User Browser
  |
  v
Frontend HTML/CSS/JavaScript
  |
  v
FastAPI Backend
  |
  +--> XGBoost Tabular Models
  |      - condition_xgb.json
  |      - risk_xgb.json
  |      - action_xgb.json
  |
  +--> Visual Lite Anomaly Model
  |      - reference.npz
  |      - threshold.json
  |
  +--> Local CSV Dataset
         - data/app/competition_inference_table.csv
```

Backend menjalankan inferensi secara sinkron. Tidak ada database server, background worker, authentication service, cloud dependency, atau automated data logging.

## Tech Stack

| Layer | Teknologi |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend API | FastAPI |
| Runtime server | Uvicorn |
| Tabular AI | XGBoost |
| Data processing | Pandas, NumPy |
| Visual anomaly | Pillow, NumPy |
| Packaging | Docker, Docker Compose |

## AI Yang Digunakan

### 1. Tabular AI

AI utama menggunakan tiga model XGBoost:

- `condition_xgb.json`: klasifikasi kondisi lot menjadi `NORMAL`, `CHECK`, atau `POOR`.
- `risk_xgb.json`: regresi skor risiko kualitas `0-100`.
- `action_xgb.json`: klasifikasi rekomendasi operasional.

Input model berasal dari fitur cold-chain dan logistik, seperti:

- `lot_mass_kg`;
- `mean_temp_c`;
- `max_temp_c`;
- `time_above_4c_h`;
- `time_above_10c_h`;
- `time_above_15c_h`;
- `remaining_quality_window_h`;
- `visual_proxy_score_0_16`;
- `handling_scenario`;
- `target_market_node_id`.

Dataset training saat ini menggunakan synthetic/weak/scenario labels dari paket AIC. Karena itu, hasil model harus dipahami sebagai MVP decision-support dan perlu dikalibrasi ulang dengan data lapangan sebelum penggunaan operasional.

### 2. Visual Anomaly

Jika pengguna mengunggah foto, sistem menjalankan visual anomaly screening. Modul ini membandingkan foto input dengan referensi foto lemuru normal menggunakan fitur warna dan tekstur sederhana.

Output visual:

- `NORMAL`: foto mirip dengan referensi normal.
- `CHECK`: foto cukup berbeda dan sebaiknya dicek.
- `ANOMALOUS`: foto berbeda signifikan dari referensi normal.

Pada MVP ini, upload foto belum mengubah risk score tabular. Foto dipakai sebagai sinyal visual tambahan, bukan classifier final ikan busuk/segar.

## Dataset

Dataset utama berasal dari paket:

```text
AIC_2026_PPN_Pengambengan_Lemuru_Logistics_Network_Dataset_v3
```

Dataset yang dipakai aplikasi web:

```text
data/app/competition_inference_table.csv
```

Model tabular dilatih dari gabungan tabel:

```text
data/raw/aic_dataset_package/01_core_network/fish_lots.csv
data/raw/aic_dataset_package/01_core_network/thermal_features.csv
data/raw/aic_dataset_package/01_core_network/structured_visual_observations.csv
data/raw/aic_dataset_package/01_core_network/decision_labels.csv
```

Catatan: dataset bersifat sintetik/weak label untuk kebutuhan MVP. Output tidak boleh diklaim sebagai validasi mutu dunia nyata tanpa kalibrasi lapangan.

## Struktur Folder Penting

```text
frontend/
  index.html
  styles.css
  app.js
  assets/

src/
  api/
  tabular/
  visual/

models/
  tabular/
  visual_lite/

data/
  app/
  raw/
  images/

outputs/
  metrics/
  plots/
  predictions/

docker-compose.yml
Dockerfile
requirements-app.txt
```

## Secret / Environment

Aplikasi ini tidak membutuhkan secret atau file environment.

Tidak diperlukan:

- API key;
- token;
- credential database;
- `.env`;
- external service.

Jika form submission meminta secret zip, isi dengan keterangan:

```text
No secret/environment file is required. The application runs fully offline/local using Docker Compose.
```

## Cara Menjalankan Dengan Docker Compose

Pastikan Docker sudah berjalan, lalu jalankan:

```bash
docker compose up --build
```

Jika menggunakan WSL dan membutuhkan permission Docker:

```bash
sudo docker compose up --build
```

Setelah server aktif, buka:

```text
http://localhost:8000
```

## Cara Menggunakan Aplikasi

### Dataset Mode

1. Buka aplikasi di browser.
2. Pilih tab `Dataset`.
3. Ketik atau pilih `Lot ID`.
4. Klik `Run AI`.
5. Lihat output `Condition`, `Risk Score`, `Recommendation`, probabilities, dan key signals.

### Manual Mode

1. Pilih tab `Manual`.
2. Isi data lot baru.
3. Upload foto jika tersedia.
4. Klik `Predict New Lot`.
5. Jika ingin menyimpan input manual ke dataset lokal, centang `Save this manual lot to local dataset`.

## Endpoint API

| Endpoint | Method | Fungsi |
|---|---|---|
| `/` | GET | Membuka frontend |
| `/api/health` | GET | Health check |
| `/api/lots` | GET | List/search lot |
| `/api/predict/{lot_id}` | GET | Prediksi satu lot dataset |
| `/api/predict/manual` | POST | Prediksi input manual dan optional foto |
| `/api/export/predictions.csv` | GET | Export semua prediksi dataset lokal |
| `/api/export/uploaded-predictions.csv` | POST | Prediksi CSV upload |

## Development Tanpa Docker

Jika ingin menjalankan langsung dari Python:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/WSL/macOS:

```bash
source .venv/bin/activate
```

Install dependency:

```bash
pip install -r requirements-app.txt
```

Jalankan server:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Buka:

```text
http://localhost:8000
```

## Training Ulang Model Tabular

Model yang sudah tersimpan di `models/tabular/` sudah dapat langsung dipakai. Jika ingin training ulang:

```bash
python -m src.tabular.train_models
```

Output training:

```text
models/tabular/action_xgb.json
models/tabular/condition_xgb.json
models/tabular/risk_xgb.json
models/tabular/metadata.json
outputs/metrics/competition_model_metrics.json
outputs/plots/confusion_matrix_action.png
outputs/plots/confusion_matrix_condition.png
```

## Build Ulang Visual Reference

Jika foto referensi normal diganti atau ditambah, jalankan:

```bash
python -m src.visual.simple_anomaly --build
```

Output:

```text
models/visual_lite/reference.npz
models/visual_lite/threshold.json
```

## Batasan MVP

- Data training masih synthetic/weak/scenario labels.
- `Condition` pada MVP sangat dipengaruhi oleh `Visual Proxy Score`.
- Foto upload belum menjadi classifier busuk/segar.
- Risk score perlu dikalibrasi ulang dengan data sensor dan inspeksi lapangan.
- Rekomendasi tetap membutuhkan validasi manusia.

## Catatan Submission

Untuk demonstrasi software-only, tampilkan terminal dan aplikasi secara bersamaan. Jalankan aplikasi dengan Docker Compose, buka browser ke `http://localhost:8000`, lalu tunjukkan alur Dataset Mode, Manual Mode, optional photo upload, dan toggle bahasa.

