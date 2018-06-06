
from flask import Blueprint, render_template

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/add_noise', methods=['POST'])
def add_noise():
    return render_template('index.html')

@main.route('/remove_noise', methods=['POST'])
def remove_noise():
    return render_template('index.html')
