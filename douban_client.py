import requests
import base64
import json
from config import DOUBAN_API_KEY
from typing import List, Dict

class DoubanVisionClient:
    def __init__(self):
        self.api_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        self.api_key = DOUBAN_API_KEY
        self.model = "doubao-1.5-vision-pro-250328"
    
    def encode_image_to_url(self, image_path):
        """将本地图片编码为数据URL格式"""
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
        
        # 获取文件扩展名来确定MIME类型
        if image_path.lower().endswith('.png'):
            mime_type = 'image/png'
        elif image_path.lower().endswith(('.jpg', '.jpeg')):
            mime_type = 'image/jpeg'
        else:
            mime_type = 'image/jpeg'  # 默认使用jpeg
        
        # 编码为base64并构建数据URL
        base64_image = base64.b64encode(image_data).decode('utf-8')
        return f"data:{mime_type};base64,{base64_image}"
    
    def extract_questions_from_image(self, image_path):
        """从图片中提取题目信息"""
        try:
            # 从notepad中获取的提示词
            prompt = """你是一个专业的教育内容分析助手。用户提供了一段内容或图片，请你识别其中包含的"独立问题"。

#### **任务要求：**

1.  **识别"独立问题"**：识别内容中所有具备**完整题干、能独立解答**的"独立问题"。即使题干中包含多个小问，也只算作一个问题。
2.  **忽略非问题内容**：忽略任何不构成完整独立问题的内容，例如闲聊、命令或不完整的题干（如"设f(x)="）。
3.  **提取题干**：提取每个独立问题的完整题干，并保持原始文本表达。
4.  **生成JSON格式输出**：将识别出的问题以JSON格式输出，其中包含一个名为`questions`的数组。

#### **输出格式：**

```json
{
  "questions_count": N,
  "questions": [
    {
      "question_number": 1,
      "question_text": "题干内容1"
    },
    {
      "question_number": 2,
      "question_text": "题干内容2"
    }
  ]
}
```

请严格按照上述JSON格式输出，不要添加任何其他内容。"""
            
            # 编码图片
            image_data_url = self.encode_image_to_url(image_path)
            
            # 构建请求数据
            data = {
                "model": self.model,
                "messages": [
                    {
                        "content": [
                            {
                                "image_url": {
                                    "url": image_data_url
                                },
                                "type": "image_url"
                            },
                            {
                                "text": prompt,
                                "type": "text"
                            }
                        ],
                        "role": "user"
                    }
                ]
            }
            
            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # 发送请求
            response = requests.post(self.api_url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                
                # 提取模型返回的内容
                choices = result.get('choices', [])
                if choices:
                    content = choices[0].get('message', {}).get('content', '')
                    
                    # 尝试解析JSON响应
                    try:
                        # 寻找JSON内容（可能包含在代码块中）
                        import re
                        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                        if json_match:
                            json_content = json_match.group(1)
                        else:
                            # 如果没有代码块，尝试直接解析
                            json_content = content
                        
                        # 清理和修正JSON内容
                        json_content = self._clean_json_content(json_content)
                        
                        questions_data = json.loads(json_content)
                        
                        return {
                            'success': True,
                            'questions': questions_data.get('questions', []),
                            'questions_count': questions_data.get('questions_count', 0),
                            'raw_response': content
                        }
                    except json.JSONDecodeError as e:
                        # 如果JSON解析失败，尝试从原始内容中提取题目信息
                        try:
                            fallback_questions = self._extract_questions_from_raw_text(content)
                            if fallback_questions:
                                return {
                                    'success': True,
                                    'questions': fallback_questions,
                                    'questions_count': len(fallback_questions),
                                    'raw_response': content,
                                    'note': '使用后备解析方法'
                                }
                        except:
                            pass
                        
                        return {
                            'success': False,
                            'error': 'JSON解析失败',
                            'message': f'无法解析模型返回的JSON: {str(e)}',
                            'raw_response': content
                        }
                else:
                    return {
                        'success': False,
                        'error': '模型响应格式错误',
                        'message': '未找到有效的响应内容'
                    }
            else:
                return {
                    'success': False,
                    'error': f'API调用失败: {response.status_code}',
                    'message': response.text
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': '调用豆包API时出现错误',
                'message': str(e)
            }
    
    def _clean_json_content(self, json_content: str) -> str:
        """清理和修正JSON内容中的特殊字符和转义问题"""
        import re
        
        # 移除可能的前后空白
        json_content = json_content.strip()
        
        # 处理JSON字符串中的特殊字符，但保留数学公式
        # 首先转义双引号（除了JSON结构中的）
        import json as json_module
        
        # 尝试先解析看是否已经是有效JSON
        try:
            json_module.loads(json_content)
            return json_content  # 如果已经是有效JSON，直接返回
        except:
            pass
        
        # 如果不是有效JSON，进行清理
        # 处理字符串值中的特殊字符
        def clean_json_string_value(match):
            key = match.group(1)
            value = match.group(2)
            
            # 转义字符串值中的特殊字符
            value = value.replace('\\', '\\\\')  # 转义反斜杠
            value = value.replace('"', '\\"')    # 转义双引号
            value = value.replace('\n', '\\n')   # 转义换行
            value = value.replace('\r', '\\r')   # 转义回车
            value = value.replace('\t', '\\t')   # 转义制表符
            
            return f'"{key}": "{value}"'
        
        # 匹配 "key": "value" 格式并清理value部分
        json_content = re.sub(r'"([^"]+)":\s*"([^"]*(?:\\.[^"]*)*)"', clean_json_string_value, json_content)
        
        # 清理多余的空格和换行
        json_content = re.sub(r'\s+', ' ', json_content)
        json_content = re.sub(r'\s*,\s*', ', ', json_content)
        json_content = re.sub(r'\s*:\s*', ': ', json_content)
        
        return json_content
    
    def _extract_questions_from_raw_text(self, content: str) -> List[Dict]:
        """从原始文本中提取题目信息作为后备方案"""
        import re
        
        questions = []
        
        # 首先尝试解析JSON中的题目信息
        json_match = re.search(r'"question_number":\s*(\d+),\s*"question_text":\s*"([^"]+)"', content)
        if json_match:
            # 找到所有JSON格式的题目
            json_matches = re.findall(r'"question_number":\s*(\d+),\s*"question_text":\s*"([^"]+)"', content)
            for match in json_matches:
                question_num = int(match[0])
                question_text = match[1].strip()
                
                # 清理题目文本中的转义字符
                question_text = self._clean_question_text(question_text)
                
                if len(question_text) > 5:
                    questions.append({
                        'question_number': question_num,
                        'question_text': question_text
                    })
            
            if questions:
                return questions
        
        # 如果JSON解析失败，尝试其他模式
        # 模式1: 题目数字 + 内容
        pattern1 = r'"question_number":\s*(\d+)[^"]*"question_text":\s*"([^"]*)"'
        matches1 = re.findall(pattern1, content, re.DOTALL)
        
        for match in matches1:
            question_num = int(match[0])
            question_text = match[1].strip()
            question_text = self._clean_question_text(question_text)
            
            if len(question_text) > 5:
                questions.append({
                    'question_number': question_num,
                    'question_text': question_text
                })
        
        if questions:
            return questions
        
        # 最后的后备方案：寻找问号或句号结尾的题目
        pattern2 = r'(\d+)[、.]?\s*([^。！？\n]+[。！？_]+)'
        matches2 = re.findall(pattern2, content)
        
        for match in matches2:
            question_num = int(match[0])
            question_text = match[1].strip()
            question_text = self._clean_question_text(question_text)
            
            if len(question_text) > 10:
                questions.append({
                    'question_number': question_num,
                    'question_text': question_text
                })
        
        return questions
    
    def _clean_question_text(self, text: str) -> str:
        """清理题目文本中的特殊字符"""
        import re
        
        # 移除多余的转义字符
        text = re.sub(r'\\+', r'\\', text)
        
        # 处理LaTeX公式
        text = re.sub(r'\$([^$]+)\$', r'(\1)', text)
        text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', text)
        text = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', text)
        text = re.sub(r'\\triangle', 'triangle', text)
        text = re.sub(r'\\odot', 'circle', text)
        text = re.sub(r'\\angle', 'angle', text)
        text = re.sub(r'\\perp', 'perp', text)
        text = re.sub(r'\\circ', 'deg', text)
        
        # 清理多余空格
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text 