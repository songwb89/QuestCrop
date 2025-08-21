import base64
import json
import requests
from PIL import Image
import io
import time
from xunfei_auth import XunfeiAuth
from config import XUNFEI_APPID

class XunfeiOCRClient:
    def __init__(self):
        self.auth = XunfeiAuth()
        self.appid = XUNFEI_APPID
        self.max_retries = 3
        self.retry_delay = 2  # 秒
    
    def encode_image(self, image_path):
        """将图片编码为base64"""
        try:
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                
            # 检查文件大小（base64编码后不超过4M）
            if len(image_data) > 3 * 1024 * 1024:  # 预留编码空间
                print(f"图片文件过大 ({len(image_data)} bytes)，正在压缩...")
                # 压缩图片
                image = Image.open(io.BytesIO(image_data))
                # 计算新尺寸
                max_size = (1024, 1024)
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # 保存压缩后的图片到内存
                output = io.BytesIO()
                image.save(output, format='JPEG', quality=85)
                image_data = output.getvalue()
                print(f"图片已压缩到 {len(image_data)} bytes")
            
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            print(f"图片编码失败: {str(e)}")
            raise
    
    def recognize_image(self, image_path):
        """识别图片中的文字和图形元素"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                print(f"尝试第 {attempt + 1} 次OCR识别...")
                
                # 获取鉴权URL和日期
                auth_url, date = self.auth.get_auth_url()
                print(f"使用认证URL: {auth_url}")
                
                # 编码图片
                image_base64 = self.encode_image(image_path)
                print(f"图片编码完成，长度: {len(image_base64)} 字符")
                
                # 构建请求数据
                data = {
                    "header": {
                        "app_id": self.appid,
                        "uid": "39769795890",
                        "did": "SR082321940000200",
                        "imei": "8664020318693660",
                        "imsi": "4600264952729100",
                        "mac": "6c:92:bf:65:c6:14",
                        "net_type": "wifi",
                        "net_isp": "CMCC",
                        "status": 0,
                        "request_id": None,
                        "res_id": ""
                    },
                    "parameter": {
                        "ocr": {
                            "result_option": "normal",
                            "result_format": "json",
                            "output_type": "one_shot",
                            "exif_option": "0",
                            "json_element_option": "",
                            "markdown_element_option": "watermark=0,page_header=0,page_footer=0,page_number=0,graph=1",
                            "sed_element_option": "watermark=0,page_header=0,page_footer=0,page_number=0,graph=1",
                            "alpha_option": "0",
                            "rotation_min_angle": 5,
                            "result": {
                                "encoding": "utf8",
                                "compress": "raw",
                                "format": "plain"
                            }
                        }
                    },
                    "payload": {
                        "image": {
                            "encoding": "jpg",
                            "image": image_base64,
                            "status": 0,
                            "seq": 0
                        }
                    }
                }
                
                # 获取请求头
                headers = self.auth.get_headers(date)
                
                # 发送请求
                print("发送OCR请求...")
                response = requests.post(auth_url, headers=headers, json=data, timeout=30)
                
                print(f"API响应状态码: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # 检查API返回的错误代码
                    if 'header' in result and 'code' in result['header']:
                        api_code = result['header']['code']
                        if api_code != 0:
                            error_msg = result['header'].get('message', f'未知错误，代码: {api_code}')
                            print(f"API返回错误代码: {api_code}, 消息: {error_msg}")
                            
                            # 特殊处理1004错误
                            if api_code == 1004:
                                error_detail = self._get_error_1004_detail(result)
                                print(f"错误1004详情: {error_detail}")
                                if attempt < self.max_retries - 1:
                                    print(f"等待 {self.retry_delay} 秒后重试...")
                                    time.sleep(self.retry_delay)
                                    continue
                            
                            return {
                                'success': False,
                                'error': f'API错误 {api_code}: {error_msg}',
                                'message': f'讯飞OCR服务返回错误代码 {api_code}'
                            }
                    
                    # 解析成功的响应
                    return self.parse_response(result)
                else:
                    error_msg = f'HTTP {response.status_code}: {response.text}'
                    print(f"HTTP错误: {error_msg}")
                    last_error = error_msg
                    
            except requests.exceptions.Timeout:
                error_msg = "请求超时"
                print(f"请求超时 (第 {attempt + 1} 次尝试)")
                last_error = error_msg
                
            except requests.exceptions.ConnectionError:
                error_msg = "网络连接错误"
                print(f"网络连接错误 (第 {attempt + 1} 次尝试)")
                last_error = error_msg
                
            except Exception as e:
                error_msg = f"处理过程中出现错误: {str(e)}"
                print(f"异常错误: {error_msg}")
                last_error = error_msg
            
            # 如果不是最后一次尝试，等待后重试
            if attempt < self.max_retries - 1:
                print(f"等待 {self.retry_delay} 秒后进行第 {attempt + 2} 次重试...")
                time.sleep(self.retry_delay)
                self.retry_delay *= 1.5  # 指数退避
        
        # 所有重试都失败了
        return {
            'success': False,
            'error': '达到最大重试次数',
            'message': f'经过 {self.max_retries} 次尝试后仍然失败。最后错误: {last_error}'
        }
    
    def _get_error_1004_detail(self, response):
        """获取1004错误的详细信息"""
        error_details = {
            1004: "服务调用失败，可能原因：1. API配额不足 2. 网络问题 3. 服务暂时不可用"
        }
        
        # 尝试从响应中获取更多信息
        if 'header' in response:
            header = response['header']
            if 'message' in header:
                return f"{error_details.get(1004, '未知错误')}: {header['message']}"
        
        return error_details.get(1004, "未知的1004错误")
    
    def parse_response(self, response):
        """解析OCR响应并提取文字和图形元素"""
        try:
            # 如果传入的是字符串，则解析为JSON；如果已经是字典，直接使用
            if isinstance(response, str):
                response = json.loads(response)
            payload = response.get('payload', {})
            result_data = payload.get('result', {})
            
            # 保存完整的响应到JSON文件
            output_file = 'ocr_response_debug.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(response, f, indent=2, ensure_ascii=False)
            print(f"DEBUG: 完整OCR响应已保存到 {output_file}")
            
            if 'text' in result_data:
                # base64解码结果
                result_text = base64.b64decode(result_data['text']).decode('utf-8')
                result_json = json.loads(result_text)
                
                # 保存解析后的JSON到文件
                parsed_output_file = 'ocr_parsed_result.json'
                with open(parsed_output_file, 'w', encoding='utf-8') as f:
                    json.dump(result_json, f, indent=2, ensure_ascii=False)
                print(f"DEBUG: 解析后的JSON结果已保存到 {parsed_output_file}")
                
                # 提取文字和图形元素
                text_elements = []
                graphic_elements = []
                
                def extract_elements(content_list, level=0):
                    """递归提取内容元素"""
                    print(f"DEBUG: extract_elements 被调用，level={level}, content_list类型: {type(content_list)}")
                    if not isinstance(content_list, list):
                        print(f"DEBUG: content_list 不是列表，返回")
                        return
                    
                    print(f"DEBUG: 处理 {len(content_list)} 个元素")
                    for i, item in enumerate(content_list):
                        if isinstance(item, list):
                            # 如果元素是列表，递归处理
                            print(f"DEBUG: 第{i}个元素是列表，递归处理")
                            extract_elements(item, level + 1)
                            continue
                        elif not isinstance(item, dict):
                            print(f"DEBUG: 第{i}个元素不是字典，跳过")
                            continue
                        
                        element_type = item.get('type', '')
                        element_id = item.get('id', '')
                        coords = item.get('coord', [])
                        text_content = item.get('text', [])
                        
                        print(f"DEBUG: 第{i}个元素 - 类型: {element_type}, ID: {element_id}, 坐标数量: {len(coords)}")
                        
                        # 处理图形元素
                        if element_type == 'graph':
                            graphic_elements.append({
                                'type': 'graph',
                                'id': element_id,
                                'coordinates': coords,
                                'bbox': self._coords_to_bbox(coords),
                                'content': f"图形元素 (ID: {element_id})"
                            })
                        
                        # 处理文本元素
                        elif element_type in ['paragraph', 'textline'] and text_content:
                            text_str = ' '.join(text_content) if isinstance(text_content, list) else str(text_content)
                            if text_str.strip():
                                text_elements.append({
                                    'text': text_str,
                                    'type': element_type,
                                    'id': element_id,
                                    'coordinates': coords,
                                    'bbox': self._coords_to_bbox(coords)
                                })
                        
                        # 递归处理嵌套内容
                        if 'content' in item:
                            extract_elements(item['content'], level + 1)
                
                # 解析页面内容 - 根据实际JSON结构
                if 'image' in result_json:
                    # 处理image结构 - image是一个列表，包含页面内容
                    print(f"DEBUG: 找到image字段，包含{len(result_json['image'])}个图像")
                    for image_item in result_json['image']:
                        if 'content' in image_item:
                            print(f"DEBUG: 处理image中的content")
                            extract_elements(image_item['content'])
                elif 'root' in result_json:
                    # 处理root结构 - root是一个列表，需要递归处理每个元素
                    for root_item in result_json['root']:
                        if 'content' in root_item:
                            extract_elements(root_item['content'])
                        else:
                            # 如果root项本身就是内容列表
                            extract_elements([root_item])
                elif 'pages' in result_json:
                    # 处理pages结构
                    for page in result_json['pages']:
                        if 'content' in page:
                            extract_elements(page['content'])
                
                return {
                    'success': True,
                    'text_elements': text_elements,
                    'graphic_elements': graphic_elements,
                    'raw_result': result_json
                }
            else:
                return {
                    'success': False,
                    'error': '无法解析识别结果',
                    'message': '响应中缺少文本数据'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': '解析响应时出现错误',
                'message': str(e)
            }
    
    def _coords_to_bbox(self, coords):
        """将坐标点转换为边界框 [x_min, y_min, x_max, y_max]"""
        if not coords or not isinstance(coords, list):
            return []
        
        try:
            x_coords = [point['x'] for point in coords if 'x' in point]
            y_coords = [point['y'] for point in coords if 'y' in point]
            
            if x_coords and y_coords:
                return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        except:
            pass
        
        return [] 