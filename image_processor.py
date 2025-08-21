from PIL import Image, ImageDraw
import os
import uuid

class ImageProcessor:
    def __init__(self, output_dir='cropped_images'):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def crop_graphic_elements(self, image_path, graphic_elements):
        """根据图形元素的坐标截取图片区域"""
        try:
            # 打开原图
            original_image = Image.open(image_path)
            image_width, image_height = original_image.size
            
            cropped_elements = []
            
            for i, element in enumerate(graphic_elements):
                bbox = element.get('bbox', [])
                if len(bbox) >= 4:
                    # bbox格式通常为 [x1, y1, x2, y2] 或 [x, y, width, height]
                    # 根据讯飞API文档，通常是 [x1, y1, x2, y2] 格式
                    x1, y1, x2, y2 = bbox[:4]
                    
                    # 确保坐标在图片范围内
                    x1 = max(0, min(x1, image_width))
                    y1 = max(0, min(y1, image_height))
                    x2 = max(x1, min(x2, image_width))
                    y2 = max(y1, min(y2, image_height))
                    
                    # 截取图形区域
                    cropped_region = original_image.crop((x1, y1, x2, y2))
                    
                    # 生成唯一文件名
                    element_type = element.get('type', 'unknown')
                    filename = f"{element_type}_{i}_{uuid.uuid4().hex[:8]}.png"
                    crop_path = os.path.join(self.output_dir, filename)
                    
                    # 保存截取的图片
                    cropped_region.save(crop_path)
                    
                    cropped_elements.append({
                        'element_info': element,
                        'cropped_image_path': crop_path,
                        'coordinates': {
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'width': x2 - x1,
                            'height': y2 - y1
                        },
                        'filename': filename
                    })
            
            return {
                'success': True,
                'cropped_elements': cropped_elements,
                'total_count': len(cropped_elements)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': '截取图形元素时出现错误',
                'message': str(e)
            }
    
    def create_annotated_image(self, image_path, graphic_elements, text_elements=None):
        """创建带标注的图片，显示识别到的区域"""
        try:
            # 打开原图
            original_image = Image.open(image_path)
            annotated_image = original_image.copy()
            draw = ImageDraw.Draw(annotated_image)
            
            # 标注图形元素（红色框）
            for i, element in enumerate(graphic_elements):
                bbox = element.get('bbox', [])
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                    # 添加标签
                    element_type = element.get('type', 'graphic')
                    draw.text((x1, y1-15), f"{element_type}_{i}", fill='red')
            
            # 标注文字元素（蓝色框）
            if text_elements:
                for i, element in enumerate(text_elements):
                    bbox = element.get('bbox', [])
                    if len(bbox) >= 4:
                        x1, y1, x2, y2 = bbox[:4]
                        draw.rectangle([x1, y1, x2, y2], outline='blue', width=1)
            
            # 保存标注图片
            annotated_filename = f"annotated_{uuid.uuid4().hex[:8]}.png"
            annotated_path = os.path.join(self.output_dir, annotated_filename)
            annotated_image.save(annotated_path)
            
            return {
                'success': True,
                'annotated_image_path': annotated_path,
                'filename': annotated_filename
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': '创建标注图片时出现错误',
                'message': str(e)
            }
    
    def cleanup_old_files(self, max_files=50):
        """清理旧的截取文件，保持文件数量在合理范围内"""
        try:
            files = [f for f in os.listdir(self.output_dir) 
                    if os.path.isfile(os.path.join(self.output_dir, f))]
            
            if len(files) > max_files:
                # 按修改时间排序，删除最旧的文件
                files_with_time = [(f, os.path.getmtime(os.path.join(self.output_dir, f))) 
                                  for f in files]
                files_with_time.sort(key=lambda x: x[1])
                
                files_to_delete = files_with_time[:len(files) - max_files]
                for filename, _ in files_to_delete:
                    os.remove(os.path.join(self.output_dir, filename))
                    
        except Exception as e:
            print(f"清理文件时出现错误: {e}") 