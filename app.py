from flask import Flask, request, jsonify, render_template
import os
import json
import mediapipe as mp
import numpy as np
from flask_cors import CORS
from PIL import Image, ImageDraw
import io
import base64
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from dotenv import load_dotenv

app = Flask(_name_)
CORS(app)

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Configura las credenciales de Google Drive desde la variable de entorno
CLIENT_SECRET_JSON = os.getenv('GOOGLE_DRIVE_CREDENTIALS')
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# ID de la carpeta donde deseas subir la imagen
FOLDER_ID = '1hUGZeEzzfLClTNQLxMbofcffNLu_4jJ0?hl'


# Inicializar el servicio de Google Drive
def obtener_servicio_drive():
    creds = service_account.Credentials.from_service_account_info(
        json.loads(CLIENT_SECRET_JSON), scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    return service

@app.route('/upload', methods=['POST'])
def detectar_Puntos_Faciales():
    if 'file' not in request.files:
        return jsonify({'error': 'No se recibió correctamente la imagen'})

    archivo = request.files['file']
    if archivo.filename == '':
        return jsonify({'error': 'No se cargó ninguna imagen'})

    # Leer el contenido de la imagen original
    imagen_original = archivo.read()  # Leer el contenido del archivo original
    archivo.seek(0)  # Reiniciar el flujo del archivo para poder usarlo nuevamente

    # Procesar la imagen para detectar puntos faciales
    image_np = np.array(Image.open(archivo).convert('RGB'))
    mp_face_mesh = mp.solutions.face_mesh

    if image_np is None:
        return jsonify({'error': 'Error al cargar la imagen'})

    # Crear una copia de la imagen para dibujar los puntos
    imagen_con_puntos = Image.fromarray(image_np)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(image_np)
        puntos_deseados = [70, 55, 285, 300, 33, 468, 133, 362, 473, 263, 4, 185, 0, 306, 17]

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if idx in puntos_deseados:
                        h, w, _ = image_np.shape
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        size = 5
                        color = (255, 0, 0)
                        thickness = 2

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