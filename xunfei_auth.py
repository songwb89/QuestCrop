import base64
import hashlib
import hmac
import json
from datetime import datetime
from urllib.parse import urlencode, urlparse
from config import XUNFEI_API_KEY, XUNFEI_API_SECRET, XUNFEI_OCR_URL

class XunfeiAuth:
    def __init__(self):
        self.api_key = XUNFEI_API_KEY
        self.api_secret = XUNFEI_API_SECRET
        self.url = XUNFEI_OCR_URL
    
    def generate_signature(self, host, date, request_line):
        """生成签名"""
        # 构建签名字符串
        signature_str = f"host: {host}\ndate: {date}\n{request_line}"
        
        # 使用HMAC-SHA256生成签名
        signature = base64.b64encode(
            hmac.new(
                self.api_secret.encode('utf-8'),
                signature_str.encode('utf-8'),
                digestmod=hashlib.sha256
            ).digest()
        ).decode('utf-8')
        
        return signature
    
    def get_auth_url(self):
        """生成带鉴权参数的URL"""
        # 解析URL
        parsed_url = urlparse(self.url)
        host = parsed_url.netloc
        path = parsed_url.path
        
        # 生成当前时间戳
        now = datetime.utcnow()
        date = now.strftime('%a, %d %b %Y %H:%M:%S GMT')
        
        # 构建请求行
        request_line = f"POST {path} HTTP/1.1"
        
        # 生成签名
        signature = self.generate_signature(host, date, request_line)
        
        # 构建authorization参数
        authorization_origin = f'api_key="{self.api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature}"'
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode('utf-8')
        
        # 构建最终URL
        auth_params = {
            'authorization': authorization,
            'date': date,
            'host': host
        }
        
        auth_url = f"{self.url}?{urlencode(auth_params)}"
        return auth_url, date
    
    def get_headers(self, date):
        """获取请求头"""
        return {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Method': 'POST',
            'Host': urlparse(self.url).netloc,
            'Date': date
        } 