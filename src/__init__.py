from flask import Flask
from flask_cors import CORS
import os
from src.config.config import Config
from dotenv import load_dotenv

# loading environment variables
load_dotenv()

# declaring flask application
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# calling the dev configuration
config = Config().dev_config

# making our application to use dev env
app.env = config.ENV


from src.controllers.user_controller import users

# register user with api blueprint
app.register_blueprint(users, url_prefix="/users")

