from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import uuid
import concurrent.futures
from werkzeug.utils import secure_filename
from ocr_client import XunfeiOCRClient
from image_processor import ImageProcessor
from douban_client import DoubanVisionClient
from question_matcher import QuestionMatcher
from config import UPLOAD_FOLDER, MAX_CONTENT_LENGTH

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
CORS(app)

# 创建必要的目录
UPLOAD_FOLDER = 'uploads'
CROPPED_FOLDER = 'cropped_images'
for folder in [UPLOAD_FOLDER, CROPPED_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传、OCR识别、题目提取和图文关联"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '没有选择文件'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'})
        
        if file and allowed_file(file.filename):
            # 生成安全的文件名，避免中文文件名问题
            original_filename = file.filename
            # 获取文件扩展名
            if '.' in original_filename:
                ext = original_filename.rsplit('.', 1)[1].lower()
            else:
                ext = 'png'  # 默认扩展名
            
            # 使用UUID生成唯一文件名，保留扩展名
            unique_filename = f"{uuid.uuid4().hex}.{ext}"
            file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(file_path)
            
            # 初始化客户端
            ocr_client = XunfeiOCRClient()
            douban_client = DoubanVisionClient()
            image_processor = ImageProcessor(CROPPED_FOLDER)
            question_matcher = QuestionMatcher()
            
            # 并行调用OCR和豆包API
            ocr_result = None
            douban_result = None
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # 提交任务
                ocr_future = executor.submit(ocr_client.recognize_image, file_path)
                douban_future = executor.submit(douban_client.extract_questions_from_image, file_path)
                
                # 获取结果
                ocr_result = ocr_future.result()
                douban_result = douban_future.result()
            
            # 检查OCR结果
            if not ocr_result['success']:
                return jsonify({
                    'success': False,
                    'error': 'OCR识别失败',
                    'message': ocr_result.get('message', '')
                })
            
            # 检查豆包API结果
            if not douban_result['success']:
                return jsonify({
                    'success': False,
                    'error': '题目提取失败',
                    'message': douban_result.get('message', ''),
                    'raw_response': douban_result.get('raw_response', '')
                })
            
            # 获取数据
            graphic_elements = ocr_result.get('graphic_elements', [])
            text_elements = ocr_result.get('text_elements', [])
            model_questions = douban_result.get('questions', [])
            
            # 打印模型返回的原始JSON数据
            print("=== 模型返回的原始JSON数据 ===")
            print(douban_result.get('raw_response', ''))
            print("=== 原始JSON数据结束 ===\n")
            
            # 截取图形元素
            crop_result = image_processor.crop_graphic_elements(file_path, graphic_elements)
            
            # 执行图文关联算法
            associations = question_matcher.build_final_associations(
                model_questions, text_elements, graphic_elements
            )
            
            # 创建标注图片
            annotated_result = image_processor.create_annotated_image(
                file_path, graphic_elements, text_elements
            )
            
            # 清理旧文件
            image_processor.cleanup_old_files()
            
            return jsonify({
                'success': True,
                'original_file': unique_filename,
                'text_elements_count': len(text_elements),
                'graphic_elements_count': len(graphic_elements),
                'questions_count': len(model_questions),
                'text_elements': text_elements,
                'graphic_elements': graphic_elements,
                'model_questions': model_questions,
                'cropped_elements': crop_result.get('cropped_elements', []),
                # 添加调试信息
                'debug_info': {
                    'ocr_success': ocr_result['success'],
                    'douban_success': douban_result['success'],
                    'ocr_raw_elements_count': len(ocr_result.get('text_elements', [])) + len(ocr_result.get('graphic_elements', [])),
                    'douban_raw_response': douban_result.get('raw_response', ''),
                    'douban_parsed_questions': douban_result.get('questions', [])
                },
                'associations': associations,
                'annotated_image': annotated_result.get('filename', ''),
                'crop_success': crop_result['success'],
                'douban_raw_response': douban_result.get('raw_response', '')
            })
        
        else:
            return jsonify({'success': False, 'error': '不支持的文件格式'})
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': '处理文件时出现错误',
            'message': str(e)
        })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """提供上传文件的访问"""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/cropped/<filename>')
def cropped_file(filename):
    """提供截取图片的访问"""
    return send_from_directory(CROPPED_FOLDER, filename)

@app.route('/api/test')
def test_api():
    """API测试接口"""
    return jsonify({
        'success': True,
        'message': '讯飞OCR Demo API 正常运行',
        'version': '1.0.0'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 