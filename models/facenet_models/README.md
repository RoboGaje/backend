# 🎯 FaceNet Models

Karena ukuran file model yang besar (>200MB), model FaceNet tidak disimpan di repository ini. 

## 📥 Download Model

Model FaceNet dapat didownload dari Google Drive:
[Download FaceNet Models](https://drive.google.com/drive/folders/1cJa4RmXEZ1U0oLc0kgoJ_MnNOemxInh7?usp=sharing)

## 📁 Struktur Direktori

Setelah download, pastikan struktur file seperti berikut:
```
facenet_models/
├── README.md                   # File ini
├── best_facenet.pth           # Model FaceNet (download dari Drive)
└── class_mapping.pkl          # Mapping class untuk face recognition
```

## ⚠️ Catatan Penting
- Model `best_facenet.pth` harus diletakkan di direktori ini
- File `class_mapping.pkl` diperlukan untuk mapping identitas wajah
- Pastikan nama file sesuai karena digunakan oleh script inference 