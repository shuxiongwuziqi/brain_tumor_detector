from api.utils import CancerDetector, create_random_temp_path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin

app = Flask(__name__)
# 导入插件
cors = CORS(app)
# 导入cancer检测器
cancer_detector = CancerDetector('./api/models/cancer_detect_model.h5')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@cross_origin()
@app.route('/upload', methods = ['POST'])
def upload():
    print(request.files)
    file = request.files['file']
    file_path = create_random_temp_path(file.filename)
    file.save(file_path)
    result = cancer_detector.predict(file_path)
    print(result)
    return jsonify({
        'code': 0,
        'msg': result['type']
    })
