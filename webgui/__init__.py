
from flask import Flask

from webgui.controllers.main import main

def create_app():

    app = Flask(__name__)

    # register our blueprints
    app.register_blueprint(main)

    return app
