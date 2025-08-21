# 讯飞OCR图形元素识别与截取Demo

这是一个基于讯飞"通用文档识别（大模型）"API的演示应用，可以识别图片中的文字和图形元素，并自动截取图形元素。

## 功能特性

- 📸 支持多种图片格式（PNG, JPG, JPEG, BMP）
- 🔍 自动识别图片中的文字和图形元素
- ✂️ 自动截取图形元素并显示坐标信息
- 📊 生成带标注的图片，标出识别区域
- 🎨 现代化的Web界面
- 📱 响应式设计，支持移动设备

## 技术栈

- **后端**: Python Flask
- **前端**: HTML5 + CSS3 + JavaScript
- **图像处理**: PIL (Pillow)
- **OCR API**: 讯飞通用文档识别（大模型）

## 安装步骤

### 1. 克隆或下载项目

```bash
git clone <项目地址>
cd QuestCrop
```

### 2. 安装Python依赖

```bash
pip install -r requirements.txt
```

### 3. 配置API密钥

编辑 `config.py` 文件，确保讯飞API的配置信息正确：

```python
XUNFEI_APPID = "your_app_id"
XUNFEI_API_SECRET = "your_api_secret"
XUNFEI_API_KEY = "your_api_key"
```

### 4. 运行应用

```bash
python app.py
```

应用将在 `http://localhost:5000` 启动。

## 使用方法

1. **打开浏览器** 访问 `http://localhost:5000`
2. **选择图片** 点击"选择图片文件"按钮上传图片
3. **开始识别** 点击"开始识别"按钮进行OCR处理
4. **查看结果** 系统会显示：
   - 识别统计信息（文字区域数量、图形元素数量等）
   - 截取的图形元素及其坐标信息
   - 带标注的原图（红框标出图形元素，蓝框标出文字区域）

## 项目结构

```
QuestCrop/
├── app.py                 # Flask主应用
├── config.py             # 配置文件
├── xunfei_auth.py        # 讯飞API鉴权模块
├── ocr_client.py         # OCR识别客户端
├── image_processor.py    # 图像处理模块
├── requirements.txt      # Python依赖
├── README.md            # 项目说明
├── templates/           # HTML模板
│   └── index.html       # 主页面
├── uploads/             # 上传文件目录（自动创建）
└── cropped_images/      # 截取图片目录（自动创建）
```

## API接口

### 文件上传和识别

- **URL**: `/upload`
- **方法**: POST
- **参数**: 
  - `file`: 图片文件（multipart/form-data）
- **返回**: JSON格式的识别结果

### 静态文件访问

- **上传文件**: `/uploads/<filename>`
- **截取图片**: `/cropped/<filename>`

## 注意事项

1. **文件大小限制**: 上传的图片base64编码后不能超过4MB
2. **支持格式**: PNG、JPG、JPEG、BMP
3. **API限制**: 请遵守讯飞API的调用频率限制
4. **网络要求**: 需要稳定的网络连接访问讯飞API

## 错误处理

应用包含完善的错误处理机制：

- 文件格式验证
- 文件大小检查
- API调用异常处理
- 图像处理错误捕获
- 友好的错误信息显示

## 自定义配置

可以在 `config.py` 中修改以下配置：

- `MAX_CONTENT_LENGTH`: 最大文件上传大小
- `UPLOAD_FOLDER`: 上传文件保存目录
- API相关配置

## 开发说明

### 添加新功能

1. 在相应的模块中添加功能代码
2. 更新路由处理函数
3. 修改前端界面和JavaScript逻辑

### 扩展API支持

可以通过修改 `ocr_client.py` 来支持其他OCR API或添加更多识别功能。

## 许可证

本项目仅用于学习和演示目的。

## 技术支持

如有问题，请参考：
- [讯飞开放平台文档](https://www.xfyun.cn/doc/words/OCRforLLM/API.html)
- Flask官方文档
- Python PIL文档 