
from flask import Flask

from appname.controllers.main import main

def create_app():

    app = Flask(__name__)

    # register our blueprints
    app.register_blueprint(main)

    return app
