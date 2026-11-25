from flask import Flask, request, jsonify
from flask_cors import CORS
import time #
from PIL import Image   # membuka dan mengubah gambar
import io   # membungkus bytes agar bisa dibaca PIL
import torch
import os
import scipy
import base64
import psutil 
import numpy as np
from transformers import (
    AutoModelForImageClassification, # kelas untuk model klasifikasi 
    AutoImageProcessor, # kelas untuk memproses gambar, pra-pemrosesan gambar(resize, normalisasi, ubah ketensor)
    AutoModelForSemanticSegmentation,   # kelas untuk model segmentasi umum
    Mask2FormerForUniversalSegmentation # kelas khusus untuk Mask2Former
)
# inisialisasi aplikasi Flask
app = Flask(__name__)
CORS(app)

current_process = psutil.Process(os.getpid())

#  Load Klasifikasi model
classification_models = {}
classification_base_path = "./classification"
classification_model_names = [
    "efficientnet-b0",
    "mobilenetv2",
    "swin-tiny",
    "resnet-50"
]

for model_name in classification_model_names:
    model_path = f"{classification_base_path}/{model_name}"
    print(f"Loading classification model: {model_name}...")
    try:
        # Load processor and model
        processor = AutoImageProcessor.from_pretrained(model_path)
        model = AutoModelForImageClassification.from_pretrained(model_path)
        
        classification_models[model_name] = {
            "processor": processor,
            "model": model
        }
        print(f"Successfully loaded {model_name}")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")

# === Load Segmentasi model ===
segmentation_models = {}
segmentation_base_path = "segmentation"
segmentation_model_names = [
    "mask2former-swin-tiny",
    "segformer-b0",
    "upernet-convnext-small",
    "upernet-swin-tiny"
]

for model_name in segmentation_model_names:
    model_path = f"{segmentation_base_path}/{model_name}"
    print(f"Loading segmentation model: {model_name}...")
    try:
        processor = AutoImageProcessor.from_pretrained(model_path)
        if model_name == "mask2former-swin-tiny":
            # Mask2Former arsitekturnya berbeda, jadi memakai kelas khusus
            model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path)
        else:
            model = AutoModelForSemanticSegmentation.from_pretrained(model_path)

        segmentation_models[model_name] = {
            "processor": processor,
            "model": model
        }
        print(f"Successfully loaded {model_name}")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")


#CLASSIFICATION (1x inferensi per permintaan) 
@app.route('/classify', methods=['POST'])
def handle_classification():
    # Validasi permintaan
    data = request.json
    if not data or 'image_data' not in data or 'model_name' not in data:
        return jsonify({"error": "Invalid request. 'image_data' and 'model_name' are required."}), 400

    image_data_base64 = data['image_data']  #gambar dalam bentuk teks base64
    model_name = data['model_name']     # nama model yang diminta laptop
    
    if model_name not in classification_models:
        return jsonify({"error": f"Model '{model_name}' not found."}), 404

    try:
        # Ubah teks base64 menjadi gambar 
        image_bytes = base64.b64decode(image_data_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ambil processor dan model yang sesuai
        processor = classification_models[model_name]["processor"]
        model = classification_models[model_name]["model"]

        current_process.cpu_percent(interval=None)
        
        # 1. Mulai Pengukuran Waktu Inferensi
        start_time = time.time()
        
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            
        end_time = time.time()
        
        # Waktu Pemrosesan Internal (hanya 1x inferensi)
        processing_time_ms = (end_time - start_time) * 1000

        # Post-processing dan hasil
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        label = model.config.id2label[predicted_class_idx]
        confidence = torch.nn.functional.softmax(logits, dim=-1)[0, predicted_class_idx].item()

        # Metrik Sistem Saat Ini (CPU/RAM)
        cpu_usage = current_process.cpu_percent(interval=None)
        ram_usage_mb = current_process.memory_info().rss / (1024 * 1024) 

        # Mengembalikan metrik ke Client (main.ipynb)
        response = {
            "status": "success",
            "data": {
                "label": label,
                "confidence": confidence,
                "model_used": model_name,
            },
            # METRIK PENTING YANG DIAMBIL KLIEN
            "stats" : {
                "cpu_usage_percent": round(cpu_usage, 2),
                "ram_usage_mb": round(ram_usage_mb, 2),
                "processing_time_ms": round(processing_time_ms, 2)
            }
        }
        return jsonify(response)

    except Exception as e:
        print(f"Error during classification: {e}")
        return jsonify({"error": str(e)}), 500

# === SEGMENTATION (1x inferensi per permintaan) ===
@app.route('/segment', methods=['POST'])
def handle_segmentation():
    # validasi permintaan
    data = request.json
    if not data or 'image_data' not in data or 'model_name' not in data:
        return jsonify({"error": "Invalid request. 'image_data' and 'model_name' are required."}), 400
    
    image_data_base64 = data['image_data']
    model_name = data['model_name']

    if model_name not in segmentation_models:
        return jsonify({"error": f"Model '{model_name}' not found."}), 404

    try:
        image_bytes = base64.b64decode(image_data_base64)
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        processor = segmentation_models[model_name]["processor"]
        model = segmentation_models[model_name]["model"]

        current_process.cpu_percent(interval=None)

        # 1. Mulai Pengukuran Waktu Inferensi
        start_time = time.time()

        inputs = processor(images=original_image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        end_time = time.time()

        # Waktu Pemrosesan Internal (hanya 1x inferensi)
        processing_time_ms = (end_time - start_time) * 1000

        # Post-processing (hanya disimpan untuk memastikan inferensi lengkap)
        if model_name == "mask2former-swin-tiny":
            seg = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[original_image.size[::-1]]
            )[0]
        else:
            logits = outputs.logits
            seg_mask = torch.nn.functional.interpolate(
                logits,
                size=original_image.size[::-1],
                mode='bilinear',
                align_corners=False
            )

        # Metrik Sistem Saat Ini (CPU/RAM)
        cpu_usage = current_process.cpu_percent(interval=None)
        ram_usage_mb = current_process.memory_info().rss / (1024 * 1024) 

        # Mengembalikan metrik ke Client (main.ipynb)
        response = {
            "status": "success",
            "data": {
                "model_used": model_name,
            },
            # METRIK PENTING YANG DIAMBIL KLIEN
            "stats" : {
                "cpu_usage_percent": round(cpu_usage, 2),
                "ram_usage_mb": round(ram_usage_mb, 2),
                "processing_time_ms": round(processing_time_ms, 2)
            }
        }
        return jsonify(response)

    except Exception as e:
        print(f"Error during segmentation: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
