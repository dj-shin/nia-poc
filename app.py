from flask import Flask, g, Response, request
from werkzeug.utils import secure_filename
import redis
import os
from pydicom import dcmread
from base64 import b64encode
from PIL import Image
import io
import numpy as np
from waitress import serve
import shutil
import zipfile
import colorsys
import random
import glob


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def create_app():
    app = Flask(__name__)
    return app


app = create_app()
app.config['UPLOAD_FOLDER'] = './dicom'


def get_redis():
    r = getattr(g, '_redis', None)
    if r is None:
        r = g._redis = redis.Redis(host='localhost', port=6379, db=0)
    return r


@app.route('/state')
def get_state():
    r = get_redis()
    lock = r.get('lock')
    return {'lock': bool(lock)}


def cleanup_storage():
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        shutil.rmtree(app.config['UPLOAD_FOLDER'])


def prepare_storage():
    cleanup_storage()
    os.mkdir(app.config['UPLOAD_FOLDER'])


@app.route('/launch', methods=['POST'])
def launch():
    r = get_redis()
    lock = r.get('lock')
    if lock:
        return Response(response='Resource Locked', status=409)
    r.set('lock', 1)    # lock session

    try:
        if 'dicom' not in request.files:
            return Response(response='No dicom file', status=400)

        f = request.files['dicom']
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))

        prepare_storage()
        f.save(fpath)
        with zipfile.ZipFile(fpath) as zf:
            extract_path = os.path.join(app.config['UPLOAD_FOLDER'], 'extract')
            os.mkdir(extract_path)
            zf.extractall(extract_path)

        # TODO
        # ==================================================================
        ct_array = dcm_to_npy(extract_path)     # 3D CT array
        segmentation = np.zeros(ct_array.shape)    # model prediction
        
        # projection
        sample_idx = np.argmax(np.sum(ct_array, axis=(0, 1)))
        original_image = ct_array[..., sample_idx]
        segmentation_image = segmentation[..., sample_idx]

        color = random_colors(1)[0]
        merged_image = apply_mask(to_rgb(original_image), segmentation_image, color)
        # ==================================================================
        
        im = Image.fromarray(merged_image.astype(np.uint8))
        img_bytes = io.BytesIO()
        im.save(img_bytes, format='PNG')
        image = b64encode(img_bytes.getvalue()).decode()

        cleanup_storage()
        return {'image': image}
    except Exception as exc:
        print(exc)
    finally:
        r.delete('lock')    # unlock


def to_rgb(gray_image):
    return np.stack([gray_image.copy() for _ in range(3)], axis=2)


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def dcm_to_npy(dcm_dir):
    dcm_list = sorted(glob.glob(dcm_dir + '/**/*.dcm'))
    ct_array = []
    for dcm_file_path in dcm_list:
        ds = dcmread(dcm_file_path)
        wc = ds.get('WindowCenter', 127.5)
        ww = ds.get('WindowWidth', 255)
        intercept = ds.get('RescaleIntercept', 0)
        slope = ds.get('RescaleSlope', 1)

        arr = np.float32(ds.pixel_array)
        arr = arr * slope + intercept
        arr = (arr - wc) / ww + 0.5;

        ct_array.append(arr)

    ct_array = np.stack(ct_array, axis=2)
    ct_array = np.clip(ct_array, 0, 1)
    ct_array = np.uint8(ct_array * 255)
    return ct_array


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=7889)
