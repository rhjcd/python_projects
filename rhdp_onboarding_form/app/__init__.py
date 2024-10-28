from flask import Flask
from dotenv import load_dotenv
import os

load_dotenv()  # This loads the variables from .env

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

from app import routes
