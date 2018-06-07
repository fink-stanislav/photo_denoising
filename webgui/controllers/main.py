
from flask import Blueprint, render_template, request, session, jsonify

from PIL import Image
import numpy as np

import time
from flask.helpers import send_file, send_from_directory
import tempfile
import denoising.image_utils as iu

main = Blueprint('main', __name__)

def _get_filename_and_ext(filename):
    if '.' in filename: 
        _split = filename.split('.')
        filename = _split[0]
        ext = _split[1]
        return filename, ext
    return filename, None

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/add_noise', methods=['POST'])
def add_noise():
    imagefile = request.files.get('img', '')

    filename, ext = _get_filename_and_ext(imagefile.filename)

    if ext is not None: 
        filepath = '{}/{}.{}'.format(tempfile.gettempdir(), filename, ext)
    else:
        filepath = '{}/{}'.format(tempfile.gettempdir(), filename)

    imagefile.save(filepath)
    session['clear_image'] = filepath

    tensor = iu.file_to_tensor(filepath)
    mask, noisy = iu.generate_noise(tensor)

    if ext is not None:
        filename = '{}_noisy.{}'.format(filename, ext)
    else:
        filename = '{}_noisy.png'.format(filename)

    filepath = '{}/{}'.format(tempfile.gettempdir(), filename)
    iu.tensor_to_file(noisy, filepath)

    json_dict = {'noisy_image': filename}

    return jsonify(json_dict)

@main.route('/remove_noise', methods=['POST'])
def remove_noise():
    return render_template('index.html')

@main.route('/get_image/<filename>')
def get_image(filename):
    filename, ext = _get_filename_and_ext(filename)
    if ext is None:
        ext = 'png'
    filepath = '{}/{}.{}'.format(tempfile.gettempdir(), filename, ext)
    mime = 'image/{}'.format(ext)

    return send_file(filepath, mimetype=mime)

@main.route('/download/<filename>', methods=['GET'])
def download(filename):
    return send_from_directory(directory=tempfile.gettempdir(), filename=filename)
