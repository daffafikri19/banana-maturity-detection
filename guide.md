# Aplikasi Deteksi Tingkat Kematangan Pisang

Aplikasi ini menggunakan model YOLO untuk mendeteksi tingkat kematangan pisang (mentah, setengah matang, matang sempurna) dari gambar yang diunggah. 

## Fitur
- Deteksi otomatis tingkat kematangan pisang berupa persentase tingkat kematangan
- Konversi label dari format LabelMe ke format YOLO
- Tampilan hasil deteksi dengan bounding boxes

## Persyaratan
- Python 3.9 atau lebih baru
- OpenCV
- YOLO V5
- LabelMe untuk labeling data

## Struktur Proyek (harus sama)
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    └── val/
        ├── image1.txt
        ├── image2.txt
        └── ...
yolo5/
├──
├── data.yaml
## Persiapan
1. **Membuat Virtual Environment**
    ```bash
    python -m venv nama_env
    (windows): venv/Scripts/activate
    (MacOS / Linux): source venv/bin/activate

2. **Install packages**
    ```bash
    pip install -r requirements.txt

## Labeling
1. **Labelme**
    ```bash
    labelme

2. Arahkan ke directory dataset

3. Konversi hasil labeling (JSON) ke format YOLO menggunakan program `yolo-converter.py`

## Training Dataset
1. Sesuaikan file data.yaml

2. **Clone repository**
   ```bash
   git clone https://github.com/ultralytics/yolov5.git yolo5
   cd yolo5

3. pip install requirements.txt

4. python train.py --batch 16 --epochs 50 --data `data.yaml` --weights yolov5s.pt --cache

5. Konversi `best.pt` ke model oonx 
   python export.py --weights runs/train/exp/weights/best.pt --batch 1 --include onnx


