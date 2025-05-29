# scoring_utils.py
import torch
import cv2
import os
import numpy as np
from PIL import Image
import clip
import pandas as pd
from tqdm import tqdm
from insightface.app import FaceAnalysis
from concurrent.futures import ThreadPoolExecutor
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from Config import Config
SIM_THRESHOLD = Config.SIM_THRESHOLD

# 初始化模型
# 本地模型路径，根据实际情况修改
local_model_path = "./models/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(local_model_path).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(local_model_path)

# 初始化模型（统一管理避免重复初始化）
clip_model_aes, preprocess_aes = clip.load("./models/ViT-L-14.pt", device=DEVICE)  # CLIP 模型
aesthetic_model = torch.nn.Linear(768, 1)  # 美学评分模型
aesthetic_model.load_state_dict(torch.load("./models/sa_0_4_vit_l_14_linear.pth", map_location=DEVICE))
aesthetic_model.to(DEVICE)
aesthetic_model.eval()
face_app = FaceAnalysis(name="buffalo_l")  # 人脸分析模型
face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

def load_image(path: str) -> np.ndarray:
    # """安全加载图像并释放资源"""
    try:
        with open(path, 'rb') as f:
            img = cv2.imdecode(np.frombuffer(f.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    except Exception as e:
        print(f"❌ 加载图像失败 {path}: {e}")
        return None
def get_image_map(image_folder: str) -> dict:
    """加载图像目录为 {文件名: 图像数组(BGR)} 字典"""
    images = {}
    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg", ".webp")):
            img_path = os.path.join(image_folder, filename)
            img = load_image(img_path)
            if img is not None:
                images[filename] = img
    return images

@torch.no_grad()
def get_aesthetic_score(image_path: str) -> float:
    """计算单张图像的美学评分"""
    try:
        image = preprocess_aes(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        image_features = clip_model_aes.encode_image(image).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return aesthetic_model(image_features).item()
    except Exception as e:
        print(f"❌ 美学评分失败 {image_path}: {e}")
        return -1.0

def compute_blur_score(image_path: str) -> float:
    """计算单张图像的清晰度评分（拉普拉斯方差）"""
    img_cv = cv2.imread(image_path)
    return cv2.Laplacian(img_cv, cv2.CV_64F).var() if img_cv is not None else -1.0

def face_similarity_tools(image1: np.ndarray, image2: np.ndarray) -> float:
    """计算两张图像的人脸最大相似度（支持多人脸）"""
    faces1 = face_app.get(image1)
    faces2 = face_app.get(image2)
    if not faces1 or not faces2:
        raise ValueError("检测到无脸图像，相似度计算失败")
    
    max_sim = -1.0
    for f1 in faces1:
        for f2 in faces2:
            if hasattr(f1, 'pose') and hasattr(f2, 'pose'):
                yaw_diff = abs(f1.pose[0] - f2.pose[0])
                pitch_diff = abs(f1.pose[1] - f2.pose[1])
                roll_diff = abs(f1.pose[2] - f2.pose[2])
                if yaw_diff > 20 or pitch_diff > 20 or roll_diff > 20:
                    print(f"⚠️ 姿态角度差异较大：yaw={yaw_diff:.1f}, pitch={pitch_diff:.1f}, roll={roll_diff:.1f}")

            emb1, emb2 = f1.embedding, f2.embedding
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            max_sim = max(max_sim, sim)
    return round(max_sim * 100, 2)

def batch_face_similarity(image_map: dict) -> tuple[list, float]:
    """批量计算图像两两间的人脸相似度"""
    total_sim, count = 0.0, 0
    results = []
    image_list = list(image_map.items())

    for i in range(len(image_list)):
        for j in range(i + 1, len(image_list)):
            (name1, img1), (name2, img2) = image_list[i], image_list[j]
            try:
                sim = face_similarity_tools(img1, img2)
                print(f"{name1} <--> {name2} 相似度: {sim}")
                results.append((name1, name2, sim))
                total_sim += sim
                count += 1
            except Exception as e:
                print(f"❌ 人脸相似度计算失败 {name1}-{name2}: {e}")

    avg_sim = round(total_sim / count, 2) if count > 0 else 0.0
    print(f"平均相似度: {avg_sim}")
    return results, avg_sim

def extract_clip_features(image_paths: List[str]) -> Tuple[List[np.ndarray], List[str]]:
    """提取图像的 CLIP 特征"""
    features = []
    valid_paths = []
    print("🔍 正在提取图像特征...")
    for path in tqdm(image_paths):
        try:
            image = Image.open(path).convert("RGB")
            inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                feat = clip_model.get_image_features(**inputs)
                feat = feat / feat.norm(p=2, dim=-1, keepdim=True)  # 归一化
                features.append(feat.cpu().numpy()[0])
                valid_paths.append(path)
        except Exception as e:
            print(f"❌ 跳过 {path}，原因：{e}")
    return features, valid_paths

def compute_similarity_and_report(features: np.ndarray, paths: List[str], threshold: float) -> List[Tuple[str, str, float]]:
    """计算相似度矩阵并输出高于阈值的图像对"""
    print("📐 正在计算图像对相似度...")
    sim_matrix = cosine_similarity(features)
    results = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            sim = sim_matrix[i][j]
            if sim > threshold:
                img1 = os.path.basename(paths[i])
                img2 = os.path.basename(paths[j])
                print(f"{img1} <--> {img2} | 相似度: {sim:.4f}")
                results.append((img1, img2, sim))
    if not results:
        print("✅ 没有发现高相似度图像对，数据集较为多样。")
    return results

def batch_clip_similarity(image_dir: str, image_map: Dict[str, np.ndarray]) -> List[Tuple[str, str, float]]:
    """
    批量计算图像之间的 CLIP 相似度，并输出高于阈值的图像对。
    返回值: [(image1, image2, similarity_score), ...]
    """
    image_paths = [os.path.join(image_dir, filename) for filename in image_map.keys()]
    features, valid_paths = extract_clip_features(image_paths)
    if not features:
        print("⚠️ 无有效图像特征，跳过相似度计算。")
        return []
    return compute_similarity_and_report(np.stack(features), valid_paths, SIM_THRESHOLD)

def process_image(image_dir: str, filename: str, reference_img: np.ndarray) -> tuple:
    """处理单张图像（计算美学分、清晰度、与参考图的相似度）"""
    try:
        img_path = os.path.join(image_dir, filename)
        aes_score = get_aesthetic_score(img_path)
        blur_score = compute_blur_score(img_path)
        sim_score = face_similarity_tools(cv2.imread(img_path), reference_img)
        return (filename, aes_score, blur_score, sim_score)
    except Exception as e:
        print(f"❌ 处理 {filename} 失败: {e}")
        return (filename, -1.0, -1.0, -1.0)

def export_csv(data: list, csv_path: str):
    """导出评分结果到CSV"""
    df = pd.DataFrame(data, columns=["文件名", "美学分", "清晰度", "人脸相似度"])
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
