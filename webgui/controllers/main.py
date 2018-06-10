
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

def _square_image_if_required(filename):
    filepath = _get_filepath(filename)
    with Image.open(filepath) as pil:
        w, h = pil.size

        size = iu.calc_preferrable_size(w, h)

        session['width'] = w
        session['height'] = h

        if w == h and w == size:
            return

        iu.resize(pil, size, size).save(filepath)

def _restore_image_size_if_required(filename):
    filepath = _get_filepath(filename)
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
    return source_filename, ext

def _save_mask(mask, filename, ext='bin'):
    result_filename = _get_filename_for_result(filename, 'mask', ext=ext)
    filepath = _get_filepath(result_filename)
    torch.save(mask, filepath)
    return result_filename

def _save_noisy(noisy, filename, ext):
    return _save_tensor_as_image(noisy, filename, 'noisy', ext)

def _save_denoised(denoised, filename, ext):
    return _save_tensor_as_image(denoised, filename, 'denoised', ext)

def _save_tensor_as_image(tensor, filename, postfix, ext):
    result_filename = _get_filename_for_result(filename, postfix, ext)
    filepath = _get_filepath(result_filename)
    tensor_image = iu.tensor_to_pil(tensor)
    tensor_image.save(filepath)
    return tensor_image, result_filename

@main.route('/add_noise', methods=['POST'])
def add_noise():
    imagefile = request.files.get('img', '')
    intencity = float(request.form['intencity']) / 100

    source_filename, ext = _save_source(imagefile)
    session['source_filename'] = '{}.{}'.format(source_filename, ext)
    _square_image_if_required(session['source_filename'])

    original = Image.open(_get_filepath(session['source_filename']))
    tensor = iu.pil_to_tensor(original)
    mask, noisy = iu.generate_noise(tensor, prop=intencity)

    session['mask'] = _save_mask(mask, source_filename)
    noisy, session['noisy'] = _save_noisy(noisy, source_filename, ext)

    psnr = iu.psnr(noisy, original)
    return jsonify({'noisy_image': session['noisy'], 'psnr': psnr})

@main.route('/remove_noise', methods=['POST'])
def remove_noise():
    steps = int(request.form['steps'])
    min_loss = int(request.form['min_loss'])

    noisy_path = _get_filepath(session['noisy'])
    noisy = iu.file_to_tensor(noisy_path)

    mask_path = _get_filepath(session['mask'])
    mask = torch.load(mask_path)

    result = Denoiser(num_steps=steps, min_loss=min_loss).denoise(mask, noisy)

    source_filename = session['source_filename']
    source_filepath = _get_filepath(source_filename)
    source_filename, ext = _get_filename_and_ext(source_filename)
    denoised, result_filename = _save_denoised(result, source_filename, ext)

    _restore_image_size_if_required(result_filename)

    original = Image.open(source_filepath)
    psnr = iu.psnr(denoised, original)
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
