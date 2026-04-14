import os
import json
import numpy as np
import onnxruntime as ort
import logging
from typing import Dict, Tuple

class PredictionService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Path ke file model dan statistik
        self.base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.model_path = os.path.join(self.base_path, "trained_models", "model_tsunami_final.onnx")
        self.stats_path = os.path.join(self.base_path, "trained_models", "normalization.json")
        
        # Inisialisasi ONNX Runtime (CPU/GPU)
        self.session = self._load_model()
        self.norm_stats = self._load_stats()
        
        # Batas Normalisasi Fault (Harus sinkron dengan Program 2)
        self.fault_bounds = {
            'mw': (7.0, 9.2), 'lon0': (103.0, 108.0), 'lat0': (-9.0, -4.5),
            'depth': (5.0, 55.0), 'strike': (260.0, 330.0), 'dip': (5.0, 40.0),
            'rake': (70.0, 115.0), 'length': (10.0, 600.0), 'width': (5.0, 250.0)
        }

    def _load_model(self):
        try:
            # Gunakan CUDA if available, else CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            return ort.InferenceSession(self.model_path, providers=providers)
        except Exception as e:
            self.logger.error(f"Gagal memuat model ONNX: {e}")
            return None

    def _load_stats(self):
        with open(self.stats_path, 'r') as f:
            return json.load(f)

    def preprocess_fault(self, p: Dict) -> np.ndarray:
        """Normalisasi Parameter Gempa ke skala 0-1 (Min-Max)"""
        vec = []
        keys = ['mw', 'lon0', 'lat0', 'depth', 'strike', 'dip', 'rake', 'length', 'width']
        for key in keys:
            low, high = self.fault_bounds[key]
            val = (p[key] - low) / (high - low)
            vec.append(np.clip(val, 0.0, 1.0))
        return np.array([vec], dtype=np.float32)

    def preprocess_dem(self, dem_patch: np.ndarray) -> np.ndarray:
        """Menghitung Slope & Shoaling + Z-Score Normalization"""
        # 1. Hitung Slope (Gradien)
        dy, dx = np.gradient(dem_patch)
        slope = np.sqrt(dx**2 + dy**2)
        
        # 2. Hitung Shoaling Proxy
        depth_sea = np.clip(-dem_patch, 1.0, None)
        shoaling = (1000.0 / depth_sea)**0.25
        
        # 3. Z-Score Normalization menggunakan stats global
        n_elev = (dem_patch - self.norm_stats['elev_mean']) / (self.norm_stats['elev_std'] + 1e-8)
        n_slope = (slope - self.norm_stats['slope_mean']) / (self.norm_stats['slope_std'] + 1e-8)
        n_shoal = (shoaling - self.norm_stats['shoal_mean']) / (self.norm_stats['shoal_std'] + 1e-8)
        
        # Stack menjadi (1, 3, 256, 256)
        input_stack = np.stack([n_elev, n_slope, n_shoal])
        return np.expand_dims(input_stack, axis=0).astype(np.float32)

    async def predict_inundation(self, fault_params: Dict, dem_patch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Menjalankan inferensi real-time.
        Input: Parameter gempa (Dict) & Patch Topografi (256x256)
        Output: Mask Bahaya (0-2) & Kedalaman Air (meter)
        """
        if self.session is None:
            raise RuntimeError("Model tidak dimuat dengan benar.")

        # 1. Preprocessing
        fault_input = self.preprocess_fault(fault_params)
        dem_input = self.preprocess_dem(dem_patch)

        # 2. ONNX Inference
        inputs = {
            self.session.get_inputs()[0].name: dem_input,
            self.session.get_inputs()[1].name: fault_input
        }
        
        outputs = self.session.run(None, inputs)
        logits, depth = outputs[0], outputs[1]

        # 3. Post-processing
        pred_mask = np.argmax(logits, axis=1).squeeze()
        pred_depth = depth.squeeze()

        # Physics-Informed Correction: Jika Mask=Aman (0), maka Depth=0
        pred_depth = np.where(pred_mask == 0, 0.0, pred_depth)

        return pred_mask, pred_depth

# Inisialisasi Singleton
prediction_service = PredictionService()
