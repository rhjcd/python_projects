import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'e89f937f8a7ff90d5806c2b3e238e6f38b615f5d80dacf8e'
    # Add other configuration settings as needed
