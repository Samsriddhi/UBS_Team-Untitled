# routes/downloads.py
import os
from flask import abort, send_file
import logging
from flask import request, jsonify
from routes import app
import numpy as np
import os
import re
from typing import List, Tuple
import logging
from flask import request, jsonify


logger = logging.getLogger(__name__)

# --- Config / logging ---
logging.basicConfig(level=logging.INFO)


from flask import Flask, request, jsonify
import re
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAYLOAD_DIR = os.path.join(BASE_DIR, "payloads")

def _serve_payload(filename: str):
    path = os.path.join(PAYLOAD_DIR, filename)
    if not os.path.isfile(path):
        abort(404)
    # Return the actual file as an attachment with the exact filename
    return send_file(
        path,
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=filename,   # Flask >= 2.0
        max_age=0
    )

@app.get("/payload_crackme")
def payload_crackme():
    return _serve_payload("payload_crackme")

@app.get("/payload_stack")
def payload_stack():
    return _serve_payload("payload_stack")

@app.get("/payload_shellcode")
def payload_shellcode():
    return _serve_payload("payload_shellcode")

@app.get("/payload_homework_mini")
def payload_homework_mini():
    return _serve_payload("payload_homework_mini")

@app.get("/payload_malicious_mini")
def payload_malicious_mini():
    return _serve_payload("payload_malicious_mini")

# Uncomment these when MD5 step is ready
# @app.get("/payload_homework")
# def payload_homework():
#     return _serve_payload("payload_homework")
#
# @app.get("/payload_malicious")
# def payload_malicious():
#     return _serve_payload("payload_malicious")
