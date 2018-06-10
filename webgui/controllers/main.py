
from flask import Blueprint, render_template, request, session, jsonify

from flask.helpers import send_file, send_from_directory
import tempfile
import denoising.image_utils as iu
import torch
from denoising.deep_image_prior import Denoiser
from PIL import Image
import time

main = Blueprint('main', __name__)

def _get_filename_and_ext(filename):
    if '.' in filename: 
        _split = filename.split('.')
        filename = _split[0]
        ext = _split[1]
        return filename, ext
    return filename, None

def _get_filepath(filename):
    filepath = '{}/{}'.format(tempfile.gettempdir(), filename)
    return filepath

def _get_filename_for_result(source_filename, result_name, ext=None):
    if ext is not None:
        filename = '{}_{}_{}.{}'.format(source_filename, result_name, _now(), ext)
    else:
        filename = '{}_{}_{}.bin'.format(source_filename, _now(), result_name)
    return filename

def _square_image_if_required(filepath):
    with Image.open(filepath) as pil:
        w, h = pil.size

        size = iu.calc_preferrable_size(w, h)

        session['width'] = w
        session['height'] = h

        if w == h and w == size:
            return

        iu.resize(pil, size, size).save(filepath)

def _restore_image_size_if_required(filepath):
    w = session['width']
    h = session['height']

    if w is None or h is None:
        return

    with Image.open(filepath) as pil:
        _w, _h = pil.size

        if w == _w and h == _h:
            return

        iu.resize(pil, w, h).save(filepath)

def _now():
    return str(int(time.time() * 100))

@main.route('/')
def home():
    return render_template('index.html')

def _save_source(imagefile):
    filepath = _get_filepath(imagefile.filename)
    source_filename, ext = _get_filename_and_ext(imagefile.filename)
    imagefile.save(filepath)
    return filepath, source_filename, ext

@main.route('/add_noise', methods=['POST'])
def add_noise():
    imagefile = request.files.get('img', '')
    intencity = float(request.form['intencity']) / 100

    filepath, source_filename, ext = _save_source(imagefile)

    _square_image_if_required(filepath)

    session['source_filename'] = '{}.{}'.format(source_filename, ext)

    original = Image.open(filepath)
    tensor = iu.pil_to_tensor(original)
    mask, noisy = iu.generate_noise(tensor, prop=intencity)

    result_filename = _get_filename_for_result(source_filename, 'mask')
    filepath = _get_filepath(result_filename)
    torch.save(mask, filepath)
    session['mask'] = filepath

    result_filename = _get_filename_for_result(source_filename, 'noisy', ext)
    filepath = _get_filepath(result_filename)
    noisy = iu.tensor_to_pil(noisy)
    noisy.save(filepath)
    session['noisy'] = filepath

    psnr = iu.psnr(noisy, original)
    return jsonify({'noisy_image': result_filename, 'psnr': psnr})

@main.route('/remove_noise', methods=['POST'])
def remove_noise():
    steps = int(request.form['steps'])
    min_loss = int(request.form['min_loss'])

    noisy_path = session['noisy']
    noisy = iu.file_to_tensor(noisy_path)

    mask_path = session['mask']
    mask = torch.load(mask_path)

    result = Denoiser(num_steps=steps, min_loss=min_loss).denoise(mask, noisy)

    source_filename = session['source_filename']
    source_filepath = _get_filepath(source_filename)
    source_filename, ext = _get_filename_and_ext(source_filename)
    result_filename = _get_filename_for_result(source_filename, 'denoised', ext)
    filepath = _get_filepath(result_filename)
    denoised_image = iu.tensor_to_pil(result)
    denoised_image.save(filepath)

    _restore_image_size_if_required(filepath)

    original = Image.open(source_filepath)
    psnr = iu.psnr(denoised_image, original)
    return jsonify({'denoised_image': result_filename, 'psnr': psnr})

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
