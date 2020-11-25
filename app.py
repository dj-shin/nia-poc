from flask import Flask, g, Response, request
from werkzeug.utils import secure_filename
import redis
import os
from pydicom import dcmread
from pydicom.pixel_data_handlers import apply_windowing, apply_rescale
from base64 import b64encode
from PIL import Image
import io
import numpy as np
from waitress import serve


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


@app.route('/launch', methods=['POST'])
def launch():
    r = get_redis()
    lock = r.get('lock')
    if lock:
        return Response(response='Resource Locked', status=409)
    r.set('lock', 1)    # lock session

    if 'dicom' not in request.files:
        r.delete('lock')    # unlock
        return Response(response='No dicom file', status=400)

    f = request.files['dicom']
    fpath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
    f.save(fpath)

    # TODO: model prediction
    image = b64encode(dcm_to_jpg(fpath)).decode()

    r.delete('lock')    # unlock
    return {'image': image}


def dcm_to_jpg(dcm_path):
    ds = dcmread(dcm_path)

    wc = ds.get('WindowCenter', 127.5)
    ww = ds.get('WindowWidth', 255)
    intercept = ds.get('RescaleIntercept', 0)
    slope = ds.get('RescaleSlope', 1)

    arr = ds.pixel_array
    arr = arr * slope + intercept
    arr = (arr - wc) * 255. / ww + 127.5;
    im = Image.fromarray(arr.astype(np.uint8))
    im = im.resize((128, 128))
    img_bytes = io.BytesIO()
    im.save(img_bytes, format='JPEG')
    return img_bytes.getvalue()


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000)
