// Import packages yang dibutuhkan
const express = require('express');
const multer = require('multer');
const cors = require('cors');
const ort = require('onnxruntime-node');
const path = require('path');
const fs = require('fs');
const cv = require('@techstark/opencv-js');
const { createCanvas, loadImage } = require('canvas');

// Inisialisasi aplikasi Express
const app = express();
app.use(cors());

// Setup Multer untuk menangani file upload di memory
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// Tentukan lokasi model
const modelPath = path.join(__dirname, 'model3.onnx');
const nmsModelPath = path.join(__dirname, 'nms.onnx');
const modelInputShape = [1, 3, 640, 640];
const topk = 100;
const iouThreshold = 0.45;
const scoreThreshold = 0.25;

// Variabel untuk menyimpan session model
let yoloSession = null;
let nmsSession = null;

// Fungsi asinkron untuk load dan "warm-up" model
async function loadModel() {
    try {
        console.log("Loading YOLO model...");
        
        // Buat session dari model yang telah diunduh dan disimpan
        yoloSession = await ort.InferenceSession.create(modelPath);
        nmsSession = await ort.InferenceSession.create(nmsModelPath);
        
        console.log("Model loaded!");

        // Warm-up model dengan tensor kosong
        console.log("Warming up the model...");
        const tensor = new ort.Tensor("float32", new Float32Array(modelInputShape.reduce((a, b) => a * b)), modelInputShape);
        await yoloSession.run({ images: tensor });
        console.log("Model warm-up completed.");
    } catch (error) {
        console.error("Error loading or warming up model:", error);
    }
}

// Panggil fungsi loadModel() saat server pertama kali dijalankan
loadModel();

app.get('/', (req, res) => {
  res.send('Success connected');
});

// Endpoint untuk menerima upload gambar dan melakukan prediksi
app.post('/upload-image', upload.single('image'), async (req, res) => {
    try {
        if (!yoloSession) {
            return res.status(500).json({ error: 'Model not loaded' });
        }

        // Ambil buffer dari file gambar yang di-upload
        const fileBuffer = req.file.buffer;
        const img = await loadImage(fileBuffer);

        const canvas = createCanvas(img.width, img.height);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, img.width, img.height);

        const detections = await detectImage(
            canvas,
            yoloSession,
            nmsSession,
            topk,
            iouThreshold,
            scoreThreshold,
            modelInputShape
        )

        res.json({ detections });

        //res.json(detections);
    } catch (error) {
        console.error("Error processing image:", error);
        res.status(500).json({ error: "Internal Server Error" });
    }
});

/**
 * Detect Image (untuk backend)
 * @param {Canvas} canvas Buffer gambar yang dikirim dari frontend
 * @param {ort.InferenceSession} yoloSession session untuk model YOLO
 * @param {ort.InferenceSession} nmsSession session untuk model NMS
 * @param {Number} topk Integer untuk jumlah maksimum kotak yang dipilih per kelas
 * @param {Number} iouThreshold Float untuk threshold IOU
 * @param {Number} scoreThreshold Float untuk threshold skor
 * @param {Number[]} inputShape Bentuk input model, biasanya [batch, channels, width, height]
 * @returns {Array} Daftar objek deteksi (label, probability, bounding box)
 */

async function detectImage (
    canvas,
    yoloSession,
    nmsSession,
    topk,
    iouThreshold,
    scoreThreshold,
    inputShape
) {
    const [modelWidth, modelHeight] = inputShape.slice(2); // Mendapatkan dimensi model

    // Proses gambar untuk mendapatkan tensor dan rasio
    const [input, xRatio, yRatio] = preprocessImage(canvas, modelWidth, modelHeight);

    // Buat tensor untuk gambar yang sudah diproses
    const tensor = new ort.Tensor('float32', input.data32F, inputShape); // Ubah data menjadi tensor

    // Tensor untuk konfigurasi NMS (topk, IOU, score threshold)
    const config = new ort.Tensor(
        'float32',
        new Float32Array([topk, iouThreshold, scoreThreshold])
    );

    try {
        const startTime = performance.now();
        // Jalankan model YOLO untuk mendapatkan output
        const { output0 } = await yoloSession.run({ images: tensor });
    
        // Jalankan NMS pada output untuk menyaring kotak-kotak deteksi
        const { selected } = await nmsSession.run({ detection: output0, config });

        const endTime = performance.now();
        const inferenceTime = endTime - startTime;
    
        const boxes = [];
    
        // Menangani hasil seleksi NMS
        for (let idx = 0; idx < selected.dims[1]; idx++) {
          const data = selected.data.slice(idx * selected.dims[2], (idx + 1) * selected.dims[2]); // Ambil baris
          const box = data.slice(0, 4);  // Bounding box
          const scores = data.slice(4);  // Probabilitas kelas
          const score = Math.max(...scores);  // Ambil skor tertinggi
          const label = scores.indexOf(score);  // Ambil kelas dengan skor tertinggi
    
          // Upscale box ke dimensi asli
          const [x, y, w, h] = [
            (box[0] - 0.5 * box[2]) * xRatio,
            (box[1] - 0.5 * box[3]) * yRatio,
            box[2] * xRatio,
            box[3] * yRatio,
          ];
    
          // Simpan hasil deteksi
          boxes.push({
            label,
            probability: score,
            bounding: [x, y, w, h],
          });
        }
    
        // Kembalikan hasil deteksi ke frontend dalam format JSON
        console.log(`Inference time: ${inferenceTime.toFixed(2)} ms`);
        console.log(boxes)
        return boxes;
    
      } catch (error) {
        console.error('Error during inference or NMS:', error);
        //throw new Error('Inference or NMS failed');
      }
}

/**
 * Preprocessing image
 * @param {Canvas} canvas buffer gambar yang di-upload
 * @param {Number} modelWidth dimensi lebar model
 * @param {Number} modelHeight dimensi tinggi model
 * @return {Array} array yang berisi tensor gambar dan rasio skala (x, y)
 */

function preprocessImage(canvas, modelWidth, modelHeight) {
    // Membaca gambar dari buffer menggunakan OpenCV
    const ctx = canvas.getContext('2d');
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const mat = cv.matFromImageData(imgData);  // Mengubah buffer menjadi gambar OpenCV
    const matC3 = new cv.Mat(mat.rows, mat.cols, cv.CV_8UC3); // Matriks untuk gambar dengan 3 channel
    cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR); // Mengubah dari RGBA ke BGR
  
    // Padding gambar agar menjadi kotak
    const maxSize = Math.max(matC3.rows, matC3.cols); // Mendapatkan ukuran maksimum dari lebar dan tinggi
    const xPad = maxSize - matC3.cols, // Padding untuk lebar
      xRatio = maxSize / matC3.cols; // Rasio skala lebar
    const yPad = maxSize - matC3.rows, // Padding untuk tinggi
      yRatio = maxSize / matC3.rows; // Rasio skala tinggi
    const matPad = new cv.Mat(); // Matriks untuk gambar yang sudah dipadding
    cv.copyMakeBorder(matC3, matPad, 0, yPad, 0, xPad, cv.BORDER_CONSTANT); // Padding dengan warna hitam
  
    // Membuat blob dari gambar untuk input model
    const input = cv.blobFromImage(
      matPad,
      1 / 255.0, // Normalisasi
      new cv.Size(modelWidth, modelHeight), // Ukuran model input
      new cv.Scalar(0, 0, 0),
      true, // SwapRB
      false // Tidak di-crop
    );
  
    // Menghapus objek OpenCV yang tidak terpakai
    mat.delete();
    matC3.delete();
    matPad.delete();
  
    return [input, xRatio, yRatio];
  };

// Menjalankan server
//const PORT = process.env.PORT || 3000;
//const HOST = '192.168.191.113'; // Ganti dengan IP address laptop kamu

/*app.listen(PORT, HOST, () => {
    console.log(`Server running on http://${HOST}:${PORT}`);
});*/

const PORT = process.env.PORT || 4040;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});