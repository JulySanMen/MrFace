from flask import Flask, request, jsonify
import os
import json
import mediapipe as mp
import numpy as np
from flask_cors import CORS
from PIL import Image, ImageEnhance, ImageDraw
import io
import base64
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Configura las credenciales de Google Drive desde la variable de entorno
CLIENT_SECRET_JSON = os.getenv('GOOGLE_DRIVE_CREDENTIALS')
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# ID de la carpeta donde deseas subir la imagen
FOLDER_ID = '1RLHKFduSGrOZNQM__5LF1HRAiduhyHMl'

# Inicializar el servicio de Google Drive
def obtener_servicio_drive():
    creds = service_account.Credentials.from_service_account_info(
        json.loads(CLIENT_SECRET_JSON), scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    return service

# Almacena la imagen original para restaurar
imagen_original_np = None

@app.route('/upload', methods=['POST'])
def detectar_Puntos_Faciales():
    global imagen_original_np
    if 'file' not in request.files:
        return jsonify({'error': 'No se recibió correctamente la imagen'})

    archivo = request.files['file']
    if archivo.filename == '':
        return jsonify({'error': 'No se cargó ninguna imagen'})

    # Leer el contenido de la imagen original
    imagen_original = archivo.read()  # Leer el contenido del archivo original
    archivo.seek(0)  # Reiniciar el flujo del archivo para poder usarlo nuevamente

    # Procesar la imagen para detectar puntos faciales
    imagen_original_np = np.array(Image.open(archivo).convert('RGB'))
    mp_face_mesh = mp.solutions.face_mesh

    if imagen_original_np is None:
        return jsonify({'error': 'Error al cargar la imagen'})

    # Crear una copia de la imagen para dibujar los puntos
    imagen_con_puntos = Image.fromarray(imagen_original_np)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(imagen_original_np)
        puntos_deseados = [70, 55, 285, 300, 33, 468, 133, 362, 473, 263, 4, 185, 0, 306, 17]

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if idx in puntos_deseados:
                        h, w, _ = imagen_original_np.shape
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        size = 10
                        color = (255, 0, 0)
                        thickness = 5

                        draw = ImageDraw.Draw(imagen_con_puntos)
                        draw.line((x - size, y - size, x + size, y + size), fill=color, width=thickness)
                        draw.line((x - size, y + size, x + size, y - size), fill=color, width=thickness)

    # Convertir la imagen procesada a formato base64
    buffered = io.BytesIO()
    imagen_con_puntos.save(buffered, format="PNG")
    img_data_con_puntos = buffered.getvalue()

    # Sube la imagen original a Google Drive manteniendo su nombre
    service = obtener_servicio_drive()
    archivo_drive = MediaIoBaseUpload(io.BytesIO(imagen_original), mimetype='image/png')
    archivo_metadata = {
        'name': archivo.filename,  # Usar el nombre original del archivo
        'mimeType': 'image/png',
        'parents': [FOLDER_ID]  # Aquí especificas la carpeta
    }
    archivo_drive_subido = service.files().create(body=archivo_metadata, media_body=archivo_drive).execute()

    return jsonify({
        'image_with_points_base64': base64.b64encode(img_data_con_puntos).decode('utf-8'),  # Imagen con puntos
        'drive_id': archivo_drive_subido.get('id')
    })

@app.route('/process', methods=['GET'])
def procesar_transformacion():
    global imagen_original_np
    if imagen_original_np is None:
        return jsonify({'error': 'No hay una imagen cargada para procesar'})

    operacion = request.args.get('operation', 'original')
    imagen = Image.fromarray(imagen_original_np)  # Crear una copia de la imagen original

    # Aplicar la operación solicitada
    if operacion == 'brightness':
        enhancer = ImageEnhance.Brightness(imagen)
        imagen = enhancer.enhance(1.5)  # Incrementar brillo
    elif operacion == 'horizontal_flip':
        imagen = imagen.transpose(Image.FLIP_LEFT_RIGHT)
    elif operacion == 'vertical_flip':
        imagen = imagen.transpose(Image.FLIP_TOP_BOTTOM)

    # Convertir la imagen transformada a formato numpy para procesar los puntos faciales
    imagen_np = np.array(imagen.convert('RGB'))  # Convertir la imagen transformada a un formato adecuado

    # Detectar y dibujar los puntos faciales en la imagen transformada
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(imagen_np)
        puntos_deseados = [70, 55, 285, 300, 33, 468, 133, 362, 473, 263, 4, 185, 0, 306, 17]

        # Crear una copia de la imagen transformada para dibujar los puntos faciales
        imagen_con_puntos = imagen.copy()
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if idx in puntos_deseados:
                        h, w, _ = imagen_np.shape
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        size = 10
                        color = (255, 0, 0)
                        thickness = 5

                        # Dibuja los puntos en la imagen transformada
                        draw = ImageDraw.Draw(imagen_con_puntos)
                        draw.line((x - size, y - size, x + size, y + size), fill=color, width=thickness)
                        draw.line((x - size, y + size, x + size, y - size), fill=color, width=thickness)

    # Convertir la imagen transformada con los puntos faciales a formato base64
    buffered = io.BytesIO()
    imagen_con_puntos.save(buffered, format="PNG")
    img_data = buffered.getvalue()

    return jsonify({
        'image_with_points_base64': base64.b64encode(img_data).decode('utf-8')
    })




if __name__ == '__main__':
    app.run(debug=True)
