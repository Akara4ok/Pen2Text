""" Web API """

import os
import sys
from flask import Flask
from flask import request
sys.path.append("Pipeline")
from pipeline_main import init_inferences
from pipeline import Pipeline
sys.path.append("Services")
from pen_text_service import PenTextService
sys.path.append("Controllers")
from pen_text_controller import PenTextController
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

word_inferences, line_inference, page_inference = init_inferences()
pipeline = Pipeline(word_inferences, line_inference, page_inference)
pen_text_service = PenTextService(pipeline)
pen_text_controller = PenTextController(pen_text_service)

@app.route('/pen_text', methods=['POST'])
def Pen2Text():
    language = request.form["language"]
    network_name = request.form["networkName"]
    uploaded_files = request.files.getlist('file')
    return pen_text_controller.process(uploaded_files, language, network_name)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)