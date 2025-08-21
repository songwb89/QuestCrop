import math
import re
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Any

class QuestionMatcher:
    def __init__(self):
        self.similarity_threshold = 0.3  # 文本相似度阈值
        self.distance_decay_factor = 150  # 距离衰减因子
        
    def calculate_euclidean_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """计算两点间的欧几里得距离"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_overlap(self, bbox1: List[float], bbox2: List[float]) -> Dict[str, float]:
        """计算两个边界框的重叠度"""
        if not bbox1 or not bbox2 or len(bbox1) < 4 or len(bbox2) < 4:
            return {'horizontal': 0, 'vertical': 0}
        
        x1_min, y1_min, x1_max, y1_max = bbox1[:4]
        x2_min, y2_min, x2_max, y2_max = bbox2[:4]
        
        # 计算水平重叠
        h_overlap_start = max(x1_min, x2_min)
        h_overlap_end = min(x1_max, x2_max)
        h_overlap = max(0, h_overlap_end - h_overlap_start)
        h_total = max(x1_max - x1_min, x2_max - x2_min)
        horizontal_overlap = h_overlap / h_total if h_total > 0 else 0
        
        # 计算垂直重叠
        v_overlap_start = max(y1_min, y2_min)
        v_overlap_end = min(y1_max, y2_max)
        v_overlap = max(0, v_overlap_end - v_overlap_start)
        v_total = max(y1_max - y1_min, y2_max - y2_min)
        vertical_overlap = v_overlap / v_total if v_total > 0 else 0
        
        return {'horizontal': horizontal_overlap, 'vertical': vertical_overlap}

    def calculate_question_territory(self, text_elements: List[Dict], model_questions: List[Dict]) -> Dict[int, Dict]:
        """为每个题目计算其势力范围（territory）"""
        text_to_question = self.match_questions_to_text_regions(model_questions, text_elements)
        
        territories = {}
        
        for question in model_questions:
            question_number = question.get('question_number', 0)
            matched_text_info = text_to_question.get(question_number)
            
            if matched_text_info:
                text_element = matched_text_info['text_element']
                text_bbox = text_element.get('bbox', [])
                
                if len(text_bbox) >= 4:
                    x1, y1, x2, y2 = text_bbox[:4]
                    
                    # 扩展势力范围：向右和向下扩展
                    # 右侧扩展：适当增加宽度用于容纳右侧图形
                    width = x2 - x1
                    expanded_x2 = x2 + width * 0.8  # 向右扩展80%的文本宽度
                    
                    # 下方扩展：适当增加高度用于容纳下方图形
                    height = y2 - y1
                    expanded_y2 = y2 + height * 2.0  # 向下扩展200%的文本高度
                    
                    territories[question_number] = {
                        'original_bbox': text_bbox,
                        'expanded_bbox': [x1, y1, expanded_x2, expanded_y2],
                        'text_element': text_element,
                        'confidence': matched_text_info.get('similarity', 0)
                    }
        
        return territories

    def is_graphic_in_territory(self, graphic_bbox: List[float], territory_bbox: List[float]) -> Dict[str, Any]:
        """判断图形是否在题目的势力范围内"""
        if not graphic_bbox or not territory_bbox or len(graphic_bbox) < 4 or len(territory_bbox) < 4:
            return {'in_territory': False, 'overlap_ratio': 0, 'position': 'unknown'}
        
        g_x1, g_y1, g_x2, g_y2 = graphic_bbox[:4]
        t_x1, t_y1, t_x2, t_y2 = territory_bbox[:4]
        
        # 计算图形中心点
        g_center_x = (g_x1 + g_x2) / 2
        g_center_y = (g_y1 + g_y2) / 2
        
        # 检查图形中心是否在势力范围内
        center_in_territory = (t_x1 <= g_center_x <= t_x2) and (t_y1 <= g_center_y <= t_y2)
        
        # 计算重叠区域
        overlap_x1 = max(g_x1, t_x1)
        overlap_y1 = max(g_y1, t_y1)
        overlap_x2 = min(g_x2, t_x2)
        overlap_y2 = min(g_y2, t_y2)
        
        if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
            overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
            graphic_area = (g_x2 - g_x1) * (g_y2 - g_y1)
            overlap_ratio = overlap_area / graphic_area if graphic_area > 0 else 0
        else:
            overlap_ratio = 0
        
        # 判断位置关系
        position = 'unknown'
        text_original_bbox = territory_bbox  # 这里简化处理，实际应该传入原始文本bbox
        
        if len(text_original_bbox) >= 4:
            text_x1, text_y1, text_x2, text_y2 = text_original_bbox[:4]
            
            # 在右侧
            if g_x1 > text_x2 and abs(g_center_y - (text_y1 + text_y2) / 2) < (text_y2 - text_y1):
                position = 'right'
            # 在下方
            elif g_y1 > text_y2 and abs(g_center_x - (text_x1 + text_x2) / 2) < (text_x2 - text_x1):
                position = 'below'
            # 在左侧
            elif g_x2 < text_x1 and abs(g_center_y - (text_y1 + text_y2) / 2) < (text_y2 - text_y1):
                position = 'left'
            # 在上方
            elif g_y2 < text_y1 and abs(g_center_x - (text_x1 + text_x2) / 2) < (text_x2 - text_x1):
                position = 'above'
        
        # 判断是否在势力范围内（任一条件满足即可）
        in_territory = center_in_territory or overlap_ratio > 0.3
        
        return {
            'in_territory': in_territory,
            'overlap_ratio': overlap_ratio,
            'center_in_territory': center_in_territory,
            'position': position
        }
    
    def calculate_spatial_relationship(self, graphic_bbox: List[float], text_bbox: List[float]) -> Dict[str, Any]:
        """计算图形与文字的空间关系和评分"""
        if not graphic_bbox or not text_bbox or len(graphic_bbox) < 4 or len(text_bbox) < 4:
            return {'score': 0, 'relationship': 'unknown', 'distance': float('inf')}
        
        g_x1, g_y1, g_x2, g_y2 = graphic_bbox[:4]
        t_x1, t_y1, t_x2, t_y2 = text_bbox[:4]
        
        # 计算中心点
        g_center = ((g_x1 + g_x2) / 2, (g_y1 + g_y2) / 2)
        t_center = ((t_x1 + t_x2) / 2, (t_y1 + t_y2) / 2)
        
        # 计算距离
        distance = self.calculate_euclidean_distance(g_center, t_center)
        
        # 计算重叠度
        overlap = self.calculate_overlap(graphic_bbox, text_bbox)
        
        # 判断位置关系
        relationship = 'unknown'
        base_score = 0
        
        # 右侧关系：图形在文字右边（试卷中最常见的布局）
        if g_x1 > t_x2 and overlap['vertical'] > 0.3:
            relationship = 'right'
            base_score = 100
        # 下方关系：图形在文字下面（试卷中也很常见）
        elif g_y1 > t_y2 and overlap['horizontal'] > 0.2:
            relationship = 'below'
            base_score = 95
        # 左侧关系：图形在文字左边（较少见）
        elif g_x2 < t_x1 and overlap['vertical'] > 0.3:
            relationship = 'left'
            base_score = 50
        # 上方关系：图形在文字上面（试卷排版中不常见，得分为0）
        elif g_y2 < t_y1 and overlap['horizontal'] > 0.2:
            relationship = 'above'
            base_score = 0
        # 内部关系：有重叠
        elif overlap['horizontal'] > 0.1 and overlap['vertical'] > 0.1:
            relationship = 'inside'
            base_score = 50
        
        # 计算最终评分（基于距离衰减）
        score = base_score * math.exp(-distance / self.distance_decay_factor)
        

        
        return {
            'score': score,
            'relationship': relationship,
            'distance': distance,
            'overlap': overlap,
            'graphic_center': g_center,
            'text_center': t_center
        }
    
    def find_best_text_for_graphic(self, graphic_element: Dict, text_elements: List[Dict]) -> Dict[str, Any]:
        """为图形元素找到最匹配的文字区域"""
        best_match = None
        best_score = 0
        
        graphic_bbox = graphic_element.get('bbox', [])
        
        for text_element in text_elements:
            text_bbox = text_element.get('bbox', [])
            
            # 计算空间关系
            spatial_result = self.calculate_spatial_relationship(graphic_bbox, text_bbox)
            
            if spatial_result['score'] > best_score:
                best_score = spatial_result['score']
                best_match = {
                    'text_element': text_element,
                    'spatial_info': spatial_result
                }
        
        return best_match if best_match else {}
    
    def normalize_text(self, text: str) -> str:
        """标准化文本，去除多余空格和特殊字符"""
        if not text:
            return ""
        
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 去除一些特殊字符，但保留基本标点
        text = re.sub(r'[^\w\s\u4e00-\u9fff\.\?\!\(\)\[\]\{\}=\+\-\*/\^]', '', text)
        
        return text.lower()
    
    def extract_keywords(self, text: str) -> List[str]:
        """提取文本中的关键词"""
        normalized = self.normalize_text(text)
        
        # 提取数字
        numbers = re.findall(r'\d+', text)
        
        # 提取中文词汇（简单按字符切分）
        chinese_words = re.findall(r'[\u4e00-\u9fff]+', text)
        
        # 提取英文单词
        english_words = re.findall(r'[a-zA-Z]+', normalized)
        
        # 合并关键词
        keywords = numbers + chinese_words + english_words
        
        # 过滤长度小于2的词
        keywords = [word for word in keywords if len(word) >= 2]
        
        return list(set(keywords))  # 去重
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度"""
        if not text1 or not text2:
            return 0.0
        
        # 标准化文本
        norm_text1 = self.normalize_text(text1)
        norm_text2 = self.normalize_text(text2)
        
        if not norm_text1 or not norm_text2:
            return 0.0
        
        # 使用SequenceMatcher计算整体相似度
        overall_similarity = SequenceMatcher(None, norm_text1, norm_text2).ratio()
        
        # 关键词匹配度
        keywords1 = set(self.extract_keywords(text1))
        keywords2 = set(self.extract_keywords(text2))
        
        if not keywords1 or not keywords2:
            keyword_similarity = 0.0
        else:
            intersection = keywords1.intersection(keywords2)
            union = keywords1.union(keywords2)
            keyword_similarity = len(intersection) / len(union) if union else 0.0
        
        # 综合评分（整体相似度权重0.4，关键词相似度权重0.6）
        final_similarity = overall_similarity * 0.4 + keyword_similarity * 0.6
        
        return final_similarity
    
    def match_questions_to_text_regions(self, model_questions: List[Dict], text_elements: List[Dict]) -> Dict[str, Any]:
        """将模型输出的题干与OCR文字区域进行匹配 - 支持多文本区域"""
        matches = {}
        used_text_elements = set()
        
        for question in model_questions:
            question_text = question.get('question_text', '')
            question_number = question.get('question_number', 0)
            
            # 收集所有可能相关的文本区域
            candidate_matches = []
            
            for i, text_element in enumerate(text_elements):
                if i in used_text_elements:
                    continue
                
                text_content = text_element.get('text', '')
                
                # 计算相似度
                similarity = self.calculate_text_similarity(question_text, text_content)
                
                # 检查是否包含题号（额外的匹配逻辑）
                contains_question_number = str(question_number) in text_content.replace(' ', '')
                
                if similarity > self.similarity_threshold or contains_question_number:
                    candidate_matches.append({
                        'text_element': text_element,
                        'text_index': i,
                        'similarity': similarity,
                        'contains_number': contains_question_number
                    })
            
            if candidate_matches:
                # 排序：优先选择相似度高的，其次选择包含题号的
                candidate_matches.sort(key=lambda x: (x['similarity'], x['contains_number']), reverse=True)
                
                # 选择最佳匹配，并收集相关区域
                best_match = candidate_matches[0]
                related_regions = [best_match]
                used_text_elements.add(best_match['text_index'])
                
                # 如果最佳匹配不包含题号，尝试找题号区域
                if not best_match['contains_number']:
                    for match in candidate_matches[1:]:
                        if match['contains_number'] and match['text_index'] not in used_text_elements:
                            related_regions.append(match)
                            used_text_elements.add(match['text_index'])
                            break
                
                matches[question_number] = {
                    'primary_match': best_match,
                    'related_regions': related_regions,
                    'all_text_elements': [m['text_element'] for m in related_regions]
                }
        
        return matches
    
    def calculate_directional_relationship(self, graphic_bbox: List[float], text_bbox: List[float]) -> Dict[str, Any]:
        """计算图形相对于文本的精确方位关系"""
        if not graphic_bbox or not text_bbox or len(graphic_bbox) < 4 or len(text_bbox) < 4:
            return {'direction': 'unknown', 'distance': float('inf'), 'alignment_score': 0}
        
        g_x1, g_y1, g_x2, g_y2 = graphic_bbox[:4]
        t_x1, t_y1, t_x2, t_y2 = text_bbox[:4]
        
        # 计算中心点
        g_center_x, g_center_y = (g_x1 + g_x2) / 2, (g_y1 + g_y2) / 2
        t_center_x, t_center_y = (t_x1 + t_x2) / 2, (t_y1 + t_y2) / 2
        
        # 计算中心点距离
        center_distance = math.sqrt((g_center_x - t_center_x)**2 + (g_center_y - t_center_y)**2)
        
        # 判断主要方位关系
        direction = 'unknown'
        alignment_score = 0
        edge_distance = float('inf')
        
        # 右侧关系：图形在文本右边
        if g_x1 > t_x2:
            direction = 'right'
            edge_distance = g_x1 - t_x2
            # 计算垂直对齐度：重叠范围越大，对齐度越高
            v_overlap_start = max(g_y1, t_y1)
            v_overlap_end = min(g_y2, t_y2)
            if v_overlap_end > v_overlap_start:
                overlap_height = v_overlap_end - v_overlap_start
                total_height = max(g_y2 - g_y1, t_y2 - t_y1)
                alignment_score = overlap_height / total_height
            
        # 下方关系：图形在文本下面
        elif g_y1 > t_y2:
            direction = 'below'
            edge_distance = g_y1 - t_y2
            # 计算水平对齐度
            h_overlap_start = max(g_x1, t_x1)
            h_overlap_end = min(g_x2, t_x2)
            if h_overlap_end > h_overlap_start:
                overlap_width = h_overlap_end - h_overlap_start
                total_width = max(g_x2 - g_x1, t_x2 - t_x1)
                alignment_score = overlap_width / total_width
        
        # 左侧关系：图形在文本左边
        elif g_x2 < t_x1:
            direction = 'left'
            edge_distance = t_x1 - g_x2
            # 计算垂直对齐度
            v_overlap_start = max(g_y1, t_y1)
            v_overlap_end = min(g_y2, t_y2)
            if v_overlap_end > v_overlap_start:
                overlap_height = v_overlap_end - v_overlap_start
                total_height = max(g_y2 - g_y1, t_y2 - t_y1)
                alignment_score = overlap_height / total_height
        
        # 上方关系：图形在文本上面
        elif g_y2 < t_y1:
            direction = 'above'
            edge_distance = t_y1 - g_y2
            # 计算水平对齐度
            h_overlap_start = max(g_x1, t_x1)
            h_overlap_end = min(g_x2, t_x2)
            if h_overlap_end > h_overlap_start:
                overlap_width = h_overlap_end - h_overlap_start
                total_width = max(g_x2 - g_x1, t_x2 - t_x1)
                alignment_score = overlap_width / total_width
        
        # 重叠关系：有交集
        else:
            direction = 'overlap'
            edge_distance = 0
            # 计算重叠面积比例
            overlap_x1 = max(g_x1, t_x1)
            overlap_y1 = max(g_y1, t_y1)
            overlap_x2 = min(g_x2, t_x2)
            overlap_y2 = min(g_y2, t_y2)
            
            if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                graphic_area = (g_x2 - g_x1) * (g_y2 - g_y1)
                alignment_score = overlap_area / graphic_area if graphic_area > 0 else 0
        
        return {
            'direction': direction,
            'center_distance': center_distance,
            'edge_distance': edge_distance,
            'alignment_score': alignment_score,
            'graphic_center': (g_center_x, g_center_y),
            'text_center': (t_center_x, t_center_y)
        }

    def calculate_graphic_assignment_score(self, graphic_bbox: List[float], text_bbox: List[float], 
                                         text_similarity: float) -> Dict[str, Any]:
        """计算图形分配给题目的综合得分"""
        directional_info = self.calculate_directional_relationship(graphic_bbox, text_bbox)
        
        direction = directional_info['direction']
        edge_distance = directional_info['edge_distance']
        alignment_score = directional_info['alignment_score']
        
        # 基础方位得分（试卷排版规律：右侧最常见，下方也常见，上方不应出现）
        direction_scores = {
            'right': 100,   # 图形在题目右边（最常见的试卷布局）
            'below': 95,    # 图形在题目下方（也很常见）
            'above': 0,     # 图形在题目上方（试卷排版中不应出现，得分为0）
            'left': 70,     # 图形在题目左边（较少见）
            'overlap': 50,  # 图形与题目重叠
            'unknown': 0    # 未知关系
        }
        
        base_score = direction_scores.get(direction, 0)
        
        # 距离惩罚：距离越远，得分越低
        if edge_distance < float('inf'):
            distance_penalty = math.exp(-edge_distance / 100)  # 距离衰减
        else:
            distance_penalty = 0
        
        # 对齐度奖励：对齐度越高，得分越高
        alignment_bonus = alignment_score * 50
        
        # 文本匹配置信度奖励
        text_confidence_bonus = text_similarity * 30
        
        # 综合得分
        final_score = (base_score * distance_penalty) + alignment_bonus + text_confidence_bonus
        
        return {
            'score': final_score,  # 前端期望的字段名
            'base_score': base_score,
            'distance_penalty': distance_penalty,
            'alignment_bonus': alignment_bonus,
            'text_confidence_bonus': text_confidence_bonus,
            'directional_info': directional_info
        }

    def find_best_graphic_assignments(self, model_questions: List[Dict], text_elements: List[Dict], 
                                    graphic_elements: List[Dict]) -> List[Dict]:
        """基于方位关系找到最佳的图形分配 - 新算法"""
        if not model_questions:
            return []
        
        # 如果没有文本元素，使用基于位置的fallback算法
        if not text_elements:
            return self._fallback_position_based_assignment(model_questions, graphic_elements)
        
        # Step 1: 文本匹配（保持原有逻辑）
        text_to_question = self.match_questions_to_text_regions(model_questions, text_elements)
        
        # Step 2: 为每个图形计算与所有题目的匹配得分
        graphic_scores = {}  # graphic_index -> [(question_number, score_info), ...]
        
        for i, graphic in enumerate(graphic_elements):
            graphic_bbox = graphic.get('bbox', [])
            if not graphic_bbox:
                continue
                
            question_scores = []
            
            for question in model_questions:
                question_number = question.get('question_number', 0)
                matched_text_info = text_to_question.get(question_number)
                
                if matched_text_info:
                    # 获取所有相关的文本区域
                    if 'all_text_elements' in matched_text_info:
                        # 新的多区域匹配
                        all_text_elements = matched_text_info['all_text_elements']
                        primary_similarity = matched_text_info['primary_match']['similarity']
                        
                        # 与每个文本区域计算关系，取最高分
                        best_score_info = None
                        best_score = 0
                        
                        for text_element in all_text_elements:
                            text_bbox = text_element.get('bbox', [])
                            
                            # 计算分配得分
                            score_info = self.calculate_graphic_assignment_score(
                                graphic_bbox, text_bbox, primary_similarity
                            )
                            
                            if score_info['score'] > best_score:
                                best_score = score_info['score']
                                best_score_info = score_info
                                best_score_info['used_text_element'] = text_element
                        
                        if best_score_info and best_score > 10:  # 最低阈值
                            question_scores.append((question_number, best_score_info))
                            
                    else:
                        # 兼容旧格式
                        text_element = matched_text_info['text_element']
                        text_bbox = text_element.get('bbox', [])
                        text_similarity = matched_text_info.get('similarity', 0)
                        
                        # 计算分配得分
                        score_info = self.calculate_graphic_assignment_score(
                            graphic_bbox, text_bbox, text_similarity
                        )
                        
                        if score_info['score'] > 10:  # 最低阈值
                            question_scores.append((question_number, score_info))
            
            # 按得分排序
            question_scores.sort(key=lambda x: x[1]['score'], reverse=True)
            graphic_scores[i] = question_scores
        
        # Step 3: 解决冲突，确保每个图形只分配给一个题目
        final_assignments = {}  # question_number -> [graphic_assignments]
        used_graphics = set()
        
        # 初始化结果结构
        for question in model_questions:
            question_number = question.get('question_number', 0)
            final_assignments[question_number] = []
        
        # 贪心分配：优先分配得分最高的组合，但考虑相对距离
        all_combinations = []
        for graphic_index, question_scores in graphic_scores.items():
            if len(question_scores) >= 2:
                # 如果有多个候选题目，检查相对距离
                first_question, first_score_info = question_scores[0]
                second_question, second_score_info = question_scores[1]
                
                # 获取两个题目的距离
                first_distance = first_score_info['directional_info']['edge_distance']
                second_distance = second_score_info['directional_info']['edge_distance']
                
                # 如果第二个题目距离明显更近（差距超过50像素），优先选择距离更近的
                if (second_distance < first_distance and 
                    first_distance - second_distance > 50 and
                    second_score_info['score'] > 30):  # 确保第二个题目得分不太低
                    question_number, score_info = second_question, second_score_info
                else:
                    question_number, score_info = first_question, first_score_info
                    
                all_combinations.append((score_info['score'], graphic_index, question_number, score_info))
                
            elif question_scores:
                # 只有一个候选题目
                question_number, score_info = question_scores[0]  # 取最高分
                all_combinations.append((score_info['score'], graphic_index, question_number, score_info))
        
        # 按得分降序排列
        all_combinations.sort(key=lambda x: x[0], reverse=True)
        
        # 执行分配
        for score, graphic_index, question_number, score_info in all_combinations:
            if graphic_index not in used_graphics:
                final_assignments[question_number].append({
                    'graphic_element': graphic_elements[graphic_index],
                    'graphic_index': graphic_index,
                    'score_info': score_info,
                    'spatial_info': score_info  # 前端期望的字段名
                })
                used_graphics.add(graphic_index)
        
        # Step 4: 构建最终结果
        final_result = []
        for question in model_questions:
            question_number = question.get('question_number', 0)
            
            # 获取文本匹配信息
            text_match_info = text_to_question.get(question_number)
            

            
            # 获取confidence值
            confidence = 0
            if text_match_info:
                if 'primary_match' in text_match_info:
                    # 新的多区域匹配格式
                    confidence = text_match_info['primary_match'].get('similarity', 0)
                else:
                    # 旧格式兼容
                    confidence = text_match_info.get('similarity', 0)
            else:
                # 当没有文本匹配信息时，使用图形匹配得分作为confidence
                graphics_assignments = final_assignments[question_number]
                if graphics_assignments:
                    # 计算图形匹配的平均得分作为confidence
                    total_score = 0
                    valid_scores = 0
                    for graphic_assignment in graphics_assignments:
                        score_info = graphic_assignment.get('score_info', {})
                        score = score_info.get('score', 0)
                        if score > 0:
                            total_score += score
                            valid_scores += 1
                    
                    if valid_scores > 0:
                        avg_score = total_score / valid_scores
                        # 将得分归一化到0-1范围，假设100分为满分
                        confidence = min(avg_score / 100.0, 1.0)
            
            final_result.append({
                'question': question,
                'question_number': question_number,
                'graphics': final_assignments[question_number],
                'text_match_info': text_match_info,
                'confidence': confidence
            })
        
        return final_result
    
    def build_final_associations(self, model_questions: List[Dict], text_elements: List[Dict], 
                                graphic_elements: List[Dict]) -> List[Dict]:
        """构建题目、文字区域和图形元素的最终关联 - 兼容接口"""
        # 调用新的基于方位的匹配算法
        return self.find_best_graphic_assignments(model_questions, text_elements, graphic_elements)

    def _fallback_position_based_assignment(self, model_questions: List[Dict], graphic_elements: List[Dict]) -> List[Dict]:
        """当没有文本元素时，基于题目顺序和图形位置的fallback分配算法"""
        if not graphic_elements:
            return []
        
        # 按Y坐标排序图形
        sorted_graphics = []
        for i, graphic in enumerate(graphic_elements):
            bbox = graphic.get('bbox', [])
            if bbox and len(bbox) >= 4:
                center_y = (bbox[1] + bbox[3]) / 2
                sorted_graphics.append((i, center_y, graphic))
        
        sorted_graphics.sort(key=lambda x: x[1])  # 按Y坐标排序
        
        # 为前几个题目分配图形
        associations = []
        num_assignments = min(len(model_questions), len(sorted_graphics))
        
        for i in range(num_assignments):
            question = model_questions[i]
            graphic_index, _, graphic_element = sorted_graphics[i]
            
            # 创建基本的关联信息
            graphic_info = {
                'graphic_index': graphic_index,
                'graphic_element': graphic_element,
                'spatial_info': {
                    'relationship': 'fallback_position',
                    'score': 50.0,  # 使用score字段而不是final_score
                    'confidence': 'low',
                    'details': f'基于位置的fallback分配 (图形{graphic_index + 1})'
                }
            }
            
            association = {
                'question_number': question.get('question_number', i + 1),
                'question_text': question.get('question_text', ''),
                'graphics': [graphic_info],
                'text_match_info': None,  # 没有文本匹配信息
                'assignment_method': 'fallback_position'
            }
            
            associations.append(association)
        
        return associations

    def _should_associate_graphic(self, question_text: str, spatial_info: Dict) -> bool:
        """智能判断是否应该将图形与题目关联"""
        score = spatial_info.get('score', 0)
        relationship = spatial_info.get('relationship', '')
        
        # 1. 如果题目明确提到图形相关词汇，降低阈值
        graphic_keywords = [
            '图', '图形', '如图', '下图', '上图', '右图', '左图',
            '三角形', '四边形', '正方形', '长方形', '圆', '扇形',
            '几何', '面积', '周长', '角度', '线段', '直线',
            '点', '弧', '半径', '直径', '切线', '弦',
            '平行', '垂直', '相似', '全等', '对称',
            '阴影', '求证', '证明'
        ]
        
        has_graphic_keywords = any(keyword in question_text for keyword in graphic_keywords)
        
        # 2. 如果题目是纯文字数学题（代数、函数等），提高阈值
        text_only_keywords = [
            '分式', '函数', '方程', '不等式', '因式分解',
            '配方', '求解', '化简', '计算', '解集',
            '定义域', '值域', '单调', '奇偶',
            '实数', '有理数', '无理数', '自然数'
        ]
        
        has_text_only_keywords = any(keyword in question_text for keyword in text_only_keywords)
        
        # 3. 根据题目类型和空间关系确定阈值
        if has_graphic_keywords:
            # 几何题，阈值较低
            threshold = 8.0
        elif has_text_only_keywords:
            # 代数题，阈值很高
            threshold = 25.0
        else:
            # 一般题目，中等阈值
            threshold = 15.0
        
        # 4. 考虑空间关系类型
        if relationship in ['inside', 'overlap']:
            # 如果是内部或重叠关系，降低阈值
            threshold *= 0.7
        elif relationship in ['far', 'very_far']:
            # 如果距离很远，提高阈值
            threshold *= 1.5
        
        return score > threshold 