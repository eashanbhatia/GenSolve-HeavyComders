import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from streamlit_drawable_canvas import st_canvas
from google.oauth2 import service_account
from googleapiclient.discovery import build
from io import BytesIO
from googleapiclient.http import MediaIoBaseUpload
import matplotlib.pyplot as plt
import requests
import os
from dotenv import load_dotenv


model = keras.models.load_model('doodle-10-2.h5')


load_dotenv()


sift = cv2.SIFT_create()
bf = cv2.BFMatcher()


class DoodlePad:
    def __init__(self, width=280, height=280):
        self.width = width
        self.height = height
        self.image = Image.new("L", (self.width, self.height), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None

    def paint(self, x, y):
        if self.last_x and self.last_y:
            self.draw.line([self.last_x, self.last_y, x, y], fill=0, width=10)
        self.last_x, self.last_y = x, y

    def reset(self):
        self.last_x, self.last_y = None, None

    def get_bitmap(self):
        resized_image = self.image.resize((28, 28))
        bitmap = np.array(resized_image)
        bitmap = 255 - bitmap  
        return bitmap


def detect_shape(bitmap):
    class_names = ['circle', 'triangle', 'moon', 'line', 'smiley_face', 'hexagon', 'square', 'octagon', 'umbrella', 'star']
    data = bitmap.reshape(28, 28, 1).astype('float32')
    pred = model.predict(np.expand_dims(data, axis=0))[0]
    ind = (-pred).argsort()[:1]
    detected_shape = [class_names[x] for x in ind][0]
    return detected_shape


def authenticate_drive():
    service_account_info = {
        "type": "service_account",
        "project_id": os.getenv("PROJECT_ID"),
        "private_key_id": os.getenv("PRIVATE_KEY_ID"),
        "private_key": os.getenv("PRIVATE_KEY").replace("\\n", "\n"),
        "client_email": os.getenv("CLIENT_EMAIL"),
        "client_id": os.getenv("CLIENT_ID"),
        "auth_uri": os.getenv("AUTH_URI"),
        "token_uri": os.getenv("TOKEN_URI"),
        "auth_provider_x509_cert_url": os.getenv("AUTH_PROVIDER_CERT_URL"),
        "client_x509_cert_url": os.getenv("CLIENT_CERT_URL"),
        "universe_domain": os.getenv("UNIVERSE_DOMAIN")
    }

    credentials = service_account.Credentials.from_service_account_info(service_account_info,
    scopes=['https://www.googleapis.com/auth/drive'])
    service = build('drive', 'v3', credentials=credentials)
    return service


def upload_to_drive(service, file, filename):
    file_metadata = {'name': filename, 'mimeType': 'image/jpeg'}
    media = MediaIoBaseUpload(file, mimetype='image/jpeg')
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    file_id = file.get('id')
    permission = {'type': 'anyone', 'role': 'reader'}
    service.permissions().create(fileId=file_id, body=permission).execute()
    file_url = f"https://drive.google.com/uc?id={file_id}"
    return file_url


class Mirror_Symmetry_detection:
    def __init__(self, image_path: str):
        self.image = self._read_color_image(image_path)
        self.reflected_image = np.fliplr(self.image)
        self.kp1, self.des1 = sift.detectAndCompute(self.image, None)
        self.kp2, self.des2 = sift.detectAndCompute(self.reflected_image, None)
    
    def _read_color_image(self, image_path: str):
        if image_path.startswith('http://') or image_path.startswith('https://'):
            response = requests.get(image_path)
            image_data = BytesIO(response.content)
            image = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image at path {image_path} could not be found or loaded.")
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        return image
    
    def find_matchpoints(self):
        matches = bf.knnMatch(self.des1, self.des2, k=2)
        matchpoints = [item[0] for item in matches]
        matchpoints = sorted(matchpoints, key=lambda x: x.distance)
        return matchpoints
    
    def find_points_r_theta(self, matchpoints):
        points_r = []
        points_theta = []
        for match in matchpoints:
            point = self.kp1[match.queryIdx]
            mirpoint = self.kp2[match.trainIdx]
            mirpoint.angle = np.deg2rad(mirpoint.angle)
            mirpoint.angle = np.pi - mirpoint.angle
            if mirpoint.angle < 0.0:
                mirpoint.angle += 2 * np.pi
            mirpoint.pt = (self.reflected_image.shape[1] - mirpoint.pt[0], mirpoint.pt[1])
            theta = angle_with_x_axis(point.pt, mirpoint.pt)
            xc, yc = midpoint(point.pt, mirpoint.pt)
            r = xc * np.cos(theta) + yc * np.sin(theta)
            points_r.append(r)
            points_theta.append(theta)
        return points_r, points_theta
    
    def find_coordinate_maxhexbin(self, image_hexbin, sorted_vote, vertical):
        for k, v in sorted_vote.items():
            if vertical:
                return k[0], k[1]
            else:
                if k[1] == 0 or k[1] == np.pi:
                    continue
                else:
                    return k[0], k[1]
    
    def sort_hexbin_by_votes(self, image_hexbin):
        counts = image_hexbin.get_array()
        ncnts = np.count_nonzero(np.power(10, counts))
        verts = image_hexbin.get_offsets()
        output = {}
        for offc in range(verts.shape[0]):
            binx, biny = verts[offc][0], verts[offc][1]
            if counts[offc]:
                output[(binx, biny)] = counts[offc]
        return {k: v for k, v in sorted(output.items(), key=lambda item: item[1], reverse=True)}
    
    def draw_mirrorLine(self, r, theta, title:str):
        mirror_line_image = np.copy(self.image)
        for y in range(len(self.image)):
            try:
                x = int((r - y * np.sin(theta)) / np.cos(theta))
                if 0 <= x < mirror_line_image.shape[1]:
                    mirror_line_image[y][x] = [255, 0, 0]
                    mirror_line_image[y][x + 1] = [255, 0, 0]
            except IndexError:
                continue
        plt.imshow(mirror_line_image)
        plt.axis('off')
        plt.title(title)
        plt.show()
        
    def check_symmetry(self, points_r):
        return len(points_r) > 0  

def angle_with_x_axis(pi, pj):
    x, y = pi[0] - pj[0], pi[1] - pj[1]
    if x == 0:
        return np.pi / 2
    angle = np.arctan(y / x)
    if angle < 0:
        angle += np.pi
    return angle

def midpoint(pi, pj):
    return (pi[0] + pj[0]) / 2, (pi[1] + pj[1]) / 2

def detecting_mirrorLine(picture_name: str, title: str, show_detail=False):
    mirror = Mirror_Symmetry_detection(picture_name)
    matchpoints = mirror.find_matchpoints()
    points_r, points_theta = mirror.find_points_r_theta(matchpoints)
    if not mirror.check_symmetry(points_r):
        plt.figure()
        plt.text(0.5, 0.5, 'No symmetry detected', fontsize=20, ha='center')
        plt.axis('off')
        plt.show()
        return
    image_hexbin = plt.hexbin(points_r, points_theta, bins=200, cmap=plt.cm.Spectral_r)
    sorted_vote = mirror.sort_hexbin_by_votes(image_hexbin)
    r, theta = mirror.find_coordinate_maxhexbin(image_hexbin, sorted_vote, vertical=False)
    mirror.draw_mirrorLine(r, theta, title)


st.title("Curvetopia - HeavyComders")

task = st.sidebar.selectbox(
    "Choose a task:",
    ("Doodle Shape Detection", "Mirror Symmetry Detection")
)

if task == "Doodle Shape Detection":
    st.subheader("Real-Time Doodle Shape Detector")
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        doodle_pad = DoodlePad()
        doodle_pad.image = Image.fromarray(canvas_result.image_data[:, :, 0].astype(np.uint8))
        bitmap = doodle_pad.get_bitmap()
        st.image(doodle_pad.image, caption="Your Drawing", use_column_width=True)
        
        if st.button("Detect Shape"):
            detected_shape = detect_shape(bitmap)
            st.write(f"Detected Shape: **{detected_shape}**")

elif task == "Mirror Symmetry Detection":
    st.subheader("Mirror Symmetry Detection")

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        if st.button('Detect Symmetry'):
            service = authenticate_drive()
            image_url = upload_to_drive(service, uploaded_file, uploaded_file.name)
            mirror = Mirror_Symmetry_detection(image_url)
            matchpoints = mirror.find_matchpoints()
            points_r, points_theta = mirror.find_points_r_theta(matchpoints)

            if not mirror.check_symmetry(points_r):
                st.text('No symmetry detected')
            else:
                fig, ax = plt.subplots()
                image_hexbin = ax.hexbin(points_r, points_theta, bins=200, cmap=plt.cm.Spectral_r)
                sorted_vote = mirror.sort_hexbin_by_votes(image_hexbin)
                r, theta = mirror.find_coordinate_maxhexbin(image_hexbin, sorted_vote, vertical=False)
                mirror.draw_mirrorLine(r, theta, "Detected Mirror Line")
                buf = BytesIO()
                plt.savefig(buf, format="png")
                st.image(buf)
