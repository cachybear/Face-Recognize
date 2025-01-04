from django.shortcuts import render
from django.shortcuts import render, get_object_or_404
import re
from django.utils import timezone
import zipfile
import joblib
import io
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
import os
from django.conf import settings
from django.http import FileResponse
import tensorflow as tf
import cv2
from django.views.decorators.csrf import csrf_exempt
import base64
from urllib.parse import urlparse
import requests
import hashlib
import xml.etree.ElementTree as ET
from django.http import JsonResponse
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from mtcnn import MTCNN
from datetime import datetime
import numpy as np
import pandas as pd
from django.utils import timezone
def home(request):
    return render(request, 'home.html')
def about_us(request):
    return render(request, 'about_us.html')

def system(request):
    return render(request, 'system.html')

# 定义保存图片的路径
UPLOAD_FOLDER = os.path.join(settings.BASE_DIR, 'myapp/model_scripts/web_photo/')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # 如果目录不存在，自动创建

@csrf_exempt
def save_photo(request):
    if request.method == 'POST':
        photo_file = request.FILES.get('photo')
        user_name = request.POST.get('userName')
        if not user_name:
            return JsonResponse({'error': '缺少用户姓名'}, status=400)
        user_folder = os.path.join(UPLOAD_FOLDER, user_name)
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)
        if  photo_file:
            file_path = os.path.join(user_folder, photo_file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in photo_file.chunks():
                    destination.write(chunk)

            return JsonResponse({'message': '照片保存成功！'}, status=200)
        return JsonResponse({'message': '没有找到照片文件'}, status=400)
    return JsonResponse({'message': '无效的请求方式'}, status=405)
#detector = MTCNN()
DATA_FOLDER = os.path.join(settings.BASE_DIR, 'myapp/model_scripts/')
@csrf_exempt
def process_and_train(request):
    if request.method == 'POST':
        try:
            image_dir = UPLOAD_FOLDER
            output_dir = os.path.join(DATA_FOLDER, 'data/')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 遍历用户文件夹（以用户姓名命名）
            for user_name in os.listdir(image_dir):
                user_input_folder = os.path.join(image_dir, user_name)  # 用户照片目录
                user_output_folder = os.path.join(output_dir, user_name)  # 输出文件夹
                if not os.path.exists(user_output_folder):
                    os.makedirs(user_output_folder)  # 创建输出目录

                extracted_count = 0  # 初始化提取人脸计数

                # 按文件名排序，确保按顺序处理
                for filename in sorted(os.listdir(user_input_folder)):
                    if extracted_count >= 10:  # 每个用户最多提取10张照片
                        break

                    name, ext = os.path.splitext(filename)
                    if ext.lower() not in ['.jpg', '.jpeg', '.png']:
                        print(f"跳过非图片文件: {filename}")
                        continue

                    # 替换 cv2.imread，处理中文路径
                    image_path = os.path.join(user_input_folder, filename)
                    try:
                        # 使用 np.fromfile 和 cv2.imdecode 加载图像
                        image_data = np.fromfile(image_path, dtype=np.uint8)
                        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                        if image is None:
                            print(f"无法读取图片: {filename}")
                            continue
                    except Exception as e:
                        print(f"读取图片失败: {filename}, 错误: {e}")
                        continue

                    # 使用 MTCNN 检测人脸
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    detector = MTCNN()
                    faces = detector.detect_faces(rgb_image)
                    if len(faces) == 0:
                        print(f"未检测到人脸: {filename}")
                        continue

                    # 提取人脸并保存
                    for result in faces:
                        if extracted_count >= 10:  # 每个用户最多提取10张照片
                            break
                        x, y, width, height = result['box']
                        face = image[y:y + height, x:x + width]
                        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        resized_face = cv2.resize(gray_face, (92, 112))
                        face_filename = os.path.join(user_output_folder, f'{extracted_count + 1}.pgm')  # 按序命名
                        cv2.imwrite(face_filename, resized_face)
                        print(f"已保存人脸: {face_filename}")
                        extracted_count += 1

                print(f"用户 {user_name} 成功提取了 {extracted_count} 张人脸图像")

            # 模型训练
            model_path, label_map_path = train_model()
            return JsonResponse({
                'message': '已完成所有用户的人脸提取与模型训练。',
                'model_path': model_path,
                'label_map_path': label_map_path
            })

        except Exception as e:
            print(f"训练过程出错: {e}")
            return JsonResponse({'error': '训练过程出错', 'details': str(e)}, status=500)

    return JsonResponse({'error': '无效的请求'}, status=400)

def load_images_from_folder(folder):
    """
    从文件夹中加载图片和标签。
    :param folder: 数据文件夹路径，每个子文件夹对应一个用户，文件夹名即为标签。
    :return: (图片数组, 标签数组, 标签映射字典)
    """
    images = []
    labels = []
    label_map = {}
    label_counter = 0
    for user_name in os.listdir(folder):
        user_folder = os.path.join(folder, user_name)
        if not os.path.isdir(user_folder):
            continue  # 跳过非文件夹的内容
        # 为用户创建一个唯一的标签 ID
        if user_name not in label_map:
            label_map[user_name] = label_counter
            label_counter += 1
        label_id = label_map[user_name]
        for filename in os.listdir(user_folder):
            if not filename.endswith('.pgm'):  # 确保文件格式正确
                continue
            img_path = os.path.join(user_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue  # 跳过无法读取的图片
            img = cv2.resize(img, (92, 112))  # 确保尺寸一致
            images.append(img.flatten())
            labels.append(label_id)
    return np.array(images), np.array(labels), label_map

import tensorflow as tf

def load_facenet_model(facenet_model_path):
    """ 加载 FaceNet 模型并返回图和会话 """
    facenet_graph = tf.Graph()
    with facenet_graph.as_default():
        facenet_sess = tf.Session()
        with facenet_sess.as_default():
            # 加载模型
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(facenet_model_path, 'rb') as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
            return facenet_graph, facenet_sess

# 修改后的 train_model 函数
def train_model():
    print("开始模型训练...")
    try:
        # 加载图片和标签
        data_folder = os.path.join(DATA_FOLDER, 'data')
        images, labels, label_map = load_images_from_folder(data_folder)
        print(f"数据集加载完成，样本数量: {len(labels)}")

        # 初始化 FaceNet 会话和图
        facenet_model_path = os.path.join(settings.BASE_DIR, 'myapp/model_scripts/20180408-102900/20180408-102900.pb')
        facenet_graph, facenet_sess = load_facenet_model(facenet_model_path)

        # 获取 FaceNet 模型的输入张量、嵌入张量和训练阶段张量
        input_tensor = facenet_graph.get_tensor_by_name('input:0')
        embeddings_tensor = facenet_graph.get_tensor_by_name('embeddings:0')
        phase_train_tensor = facenet_graph.get_tensor_by_name('phase_train:0')

        # 提取嵌入特征向量
        embeddings = []
        for img in images:
            reshaped_img = img.reshape(112, 92, 1)  # 原图像形状为 92x112，转换为 (112, 92, 1)
            resized_img = cv2.resize(reshaped_img, (160, 160))  # FaceNet 要求输入 160x160
            normalized_img = np.repeat(resized_img[..., np.newaxis], 3, axis=-1) / 255.0  # 灰度图转 RGB
            feed_dict = {input_tensor: np.expand_dims(normalized_img, axis=0), phase_train_tensor: False}
            embedding = facenet_sess.run(embeddings_tensor, feed_dict=feed_dict)
            embeddings.append(embedding.flatten())

        embeddings = np.array(embeddings)
        labels = np.array(labels)

        print("嵌入特征提取完成。开始划分数据集...")

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
        print(f"训练集大小: {len(y_train)}, 测试集大小: {len(y_test)}")

        # 训练 SVM 模型
        print("开始训练 SVM 模型...")
        svm = SVC(kernel='linear', probability=True)
        svm.fit(X_train, y_train)

        # 评估模型性能
        accuracy = svm.score(X_test, y_test)
        print(f"SVM 测试集准确率: {accuracy:.2f}")

        # 保存模型和标签映射
        model_dir = os.path.join(settings.MEDIA_ROOT, 'trained_models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(model_dir, 'svm_model.pkl')
        label_map_path = os.path.join(model_dir, 'label_map.pkl')

        joblib.dump(svm, model_path)
        joblib.dump(label_map, label_map_path)

        print(f"模型已保存至 {model_path}，标签映射已保存至 {label_map_path}")
        return model_path, label_map_path

    except Exception as e:
        print(f"模型训练失败: {e}")
        raise  # 抛出异常供调用方捕获

def load_trained_model():
    """
    加载训练好的 SVM 模型和标签映射。
    """
    model_dir = os.path.join(settings.MEDIA_ROOT, 'trained_models')
    svm_model_path = os.path.join(model_dir, 'svm_model.pkl')
    label_map_path = os.path.join(model_dir, 'label_map.pkl')

    # 加载 SVM 模型
    svm_model = joblib.load(svm_model_path)
    # 加载标签映射
    label_map = joblib.load(label_map_path)

    return svm_model, label_map
FACENET_MODEL_PATH = os.path.join(settings.BASE_DIR, 'myapp/model_scripts/20180408-102900/20180408-102900.pb')
FACENET_GRAPH, FACENET_SESS = load_facenet_model(FACENET_MODEL_PATH)
INPUT_TENSOR = FACENET_GRAPH.get_tensor_by_name('input:0')
EMBEDDINGS_TENSOR = FACENET_GRAPH.get_tensor_by_name('embeddings:0')
PHASE_TRAIN_TENSOR = FACENET_GRAPH.get_tensor_by_name('phase_train:0')

SVM_MODEL, LABEL_MAP = load_trained_model()
# 人脸检测和分类的主要 API
@csrf_exempt
def process_frame(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # 获取上传的图像
            image_file = request.FILES['image']
            image_data = np.frombuffer(image_file.read(), np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            if image is None:
                return JsonResponse({'success': False, 'error': '无法读取图像数据'})

            # 使用 MTCNN 检测人脸
            detector = MTCNN(min_face_size=10,scale_factor=0.7, steps_threshold=[0.5, 0.6, 0.7])
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb_image)
            print(len(faces))

            if len(faces) == 0:
                return JsonResponse({'success': True, 'faceData': []})  # 未检测到人脸
            results = []

            for face in faces:
                # 获取人脸框坐标
                x, y, width, height = face['box']

                # 确保坐标不超出图像范围
                x = max(0, x)
                y = max(0, y)
                width = min(width, image.shape[1] - x)
                height = min(height, image.shape[0] - y)

                face_img = image[y:y + height, x:x + width]

                # 对人脸进行预处理：灰度化、归一化、调整大小
                face_img = cv2.resize(face_img, (160, 160))  # FaceNet 需要 160x160 输入
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB) / 255.0  # 归一化
                face_img = np.expand_dims(face_img, axis=0)  # 扩展维度

                # 使用 FaceNet 提取嵌入特征
                feed_dict = {INPUT_TENSOR: face_img, PHASE_TRAIN_TENSOR: False}
                embedding = FACENET_SESS.run(EMBEDDINGS_TENSOR, feed_dict=feed_dict)

                # 使用 SVM 进行分类预测
                prediction = SVM_MODEL.predict(embedding)
                probability = SVM_MODEL.predict_proba(embedding).max()

                # 获取分类结果
                label_id = prediction[0]
                label_name = next((name for name, id_ in LABEL_MAP.items() if id_ == label_id), "Unknown")

                # 将结果添加到列表中
                results.append({
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'name': label_name,
                    'confidence': float(probability)
                })

            return JsonResponse({'success': True, 'faceData': results})

        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})

    return JsonResponse({'success': False, 'error': '无效的请求'})
