import cv2
import numpy as np
import os
import json
import base64
import requests
import random
from datetime import timedelta
import argparse
import time
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk

class MissileStrikeAnalyzer:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("无法打开视频文件")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.key_frames = {
            "missile_appearance": None,
            "missile_impact": None,
            "explosion": None,
            "explosion_end": None
        }
        self.key_frames_time = {
            "missile_appearance": None,
            "missile_impact": None,
            "explosion": None,
            "explosion_end": None
        }
        
        self.target_roi = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        self.prev_gray = None
        self.prev_points = None
        
        self.missile_trajectory = []
        self.missile_bbox = None
        
        self.explosion_threshold = 200
        self.explosion_min_area = self.frame_width * self.frame_height * 0.01
        
        self.missile_detected = False
        self.impact_detected = False
        self.explosion_detected = False
        self.explosion_end_detected = False
        
        self.frame_counter = 0
        self.impact_frame_count = 0
        self.cooldown_frames = int(self.fps * 0.02)
        
        self.previous_frames = []
        self.frames_to_keep = 5
        
        self.brightness_threshold = 30
        self.color_change_threshold = 40
        self.explosion_change_ratio_threshold = 0.1
        self.min_roi_pixels = 1024
        
        self.stability_threshold = 15
        self.stability_counter = 0
        self.pixel_stability_threshold = 5
        self.previous_explosion_frames = []
        self.explosion_frames_to_keep = 10
        self.min_contour_area_ratio = 0.02
        
        self.font_path = None
        self._init_font()
    
    def _init_font(self):
        try:
            self.font_path = "C:/Windows/Fonts/simhei.ttf"
            if not os.path.exists(self.font_path):
                self.font_path = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
                if not os.path.exists(self.font_path):
                    self.font_path = "/System/Library/Fonts/PingFang.ttc"
                    if not os.path.exists(self.font_path):
                        self.font_path = None
        except:
            self.font_path = None
    
    def select_target_roi(self):
        ret, first_frame = self.cap.read()
        if not ret:
            raise ValueError("无法读取视频的第一帧")
        
        cv2.namedWindow("Select Target ROI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select Target ROI", 800, 600)
        
        frame_with_text = self.put_chinese_text(first_frame, "请选择目标区域，按回车确认", (20, 40), 36, (0, 255, 255))
        
        self.target_roi = cv2.selectROI("Select Target ROI", frame_with_text, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Target ROI")
        
        x, y, w, h = self.target_roi
        self.target_roi = (x, y, w, h)
        print(f"已选择目标区域: ({x}, {y}), ({x+w}, {y+h})")
        return self.target_roi
    
    def check_rectangles_overlap(self, rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        if x1 >= x2 + w2 or x2 >= x1 + w1:
            return False
        if y1 >= y2 + h2 or y2 >= y1 + h1:
            return False
        return True
    
    def detect_missile(self, frame, fg_mask):
        if self.impact_detected and self.impact_frame_count < self.cooldown_frames:
            self.impact_frame_count += 1
            self.missile_bbox = None
            return frame
        
        if self.impact_detected and self.explosion_detected:
            self.missile_bbox = None
            return frame
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, moving_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
        kernel_size = max(3, frame.shape[1] // 200)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        moving_mask = cv2.morphologyEx(moving_mask, cv2.MORPH_OPEN, kernel)
        moving_mask = cv2.morphologyEx(moving_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(moving_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        frame_area = self.frame_width * self.frame_height
        min_area_abs = 100
        min_area_ratio = 0.0001
        max_area_ratio = 0.05
        min_aspect_ratio = 2.0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            if area < min_area_abs or w == 0 or h == 0:
                continue
            
            area_ratio = area / frame_area
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio < min_aspect_ratio:
                continue
            
            candidates.append({
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'center': (x + w // 2, y + h // 2)
            })
        
        best_candidate = None
        if candidates:
            if self.missile_bbox is not None:
                prev_center = (self.missile_bbox[0] + self.missile_bbox[2] // 2, 
                               self.missile_bbox[1] + self.missile_bbox[3] // 2)
                min_distance = float('inf')
                for candidate in candidates:
                    dist = np.sqrt((candidate['center'][0] - prev_center[0])**2 + 
                                  (candidate['center'][1] - prev_center[1])**2)
                    max_reasonable_dist_per_frame = (self.frame_width / 2) / self.fps if self.fps > 0 else self.frame_width / 10
                    if dist < min_distance and dist < max_reasonable_dist_per_frame * 1.5:
                        min_distance = dist
                        best_candidate = candidate
                if best_candidate:
                    self.missile_bbox = best_candidate['bbox']
                    self.missile_trajectory.append(best_candidate['center'])
                    self.missile_detected = True
            if best_candidate is None:
                best_candidate = max(candidates, key=lambda c: c['area'])
                self.missile_bbox = best_candidate['bbox']
                self.missile_trajectory.append(best_candidate['center'])
                if not self.missile_detected:
                    self.missile_detected = True
                    self.key_frames["missile_appearance"] = frame.copy()
                    self.key_frames_time["missile_appearance"] = self.frame_counter / self.fps
        
        if self.missile_bbox is not None:
            x, y, w, h = self.missile_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "导弹", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=moving_mask, **self.feature_params)
        else:
            if self.prev_points is not None and len(self.prev_points) > 0:
                next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, self.prev_points, None, **self.lk_params)
                if next_points is not None:
                    good_next = next_points[status == 1]
                    good_prev = self.prev_points[status == 1]
                    if len(good_next) >= 5:
                        self.prev_points = good_next.reshape(-1, 1, 2)
                    else:
                        self.prev_points = cv2.goodFeaturesToTrack(gray, mask=moving_mask, **self.feature_params)
                else:
                    self.prev_points = cv2.goodFeaturesToTrack(gray, mask=moving_mask, **self.feature_params)
            else:
                self.prev_points = cv2.goodFeaturesToTrack(gray, mask=moving_mask, **self.feature_params)
        
        self.prev_gray = gray.copy()
        return frame
    
    def detect_impact(self, frame):
        if self.missile_detected and not self.impact_detected and self.missile_bbox is not None:
            try:
                overlap = self.check_rectangles_overlap(self.missile_bbox, self.target_roi)
            except:
                return frame
            
            if overlap:
                self.impact_detected = True
                self.impact_frame_count = 0
                self.key_frames["missile_impact"] = frame.copy()
                self.key_frames_time["missile_impact"] = self.frame_counter / self.fps
                
                missile_x, missile_y, missile_w, missile_h = self.missile_bbox
                target_x, target_y, target_w, target_h = self.target_roi
                overlap_x = max(missile_x, target_x)
                overlap_y = max(missile_y, target_y)
                overlap_w = min(missile_x + missile_w, target_x + target_w) - overlap_x
                overlap_h = min(missile_y + missile_h, target_y + target_h) - overlap_y
                
                if overlap_w > 0 and overlap_h > 0:
                    cv2.rectangle(frame, (overlap_x, overlap_y),
                                  (overlap_x + overlap_w, overlap_y + overlap_h),
                                  (0, 255, 255), -1)
        
        if self.target_roi:
            x, y, w, h = self.target_roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "目标区域", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        if self.missile_bbox and not self.impact_detected:
            x, y, w, h = self.missile_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "导弹", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def detect_explosion(self, frame):
        if self.explosion_detected:
            return frame
        
        if not self.missile_detected:
            if len(self.previous_frames) >= self.frames_to_keep:
                self.previous_frames.pop(0)
            self.previous_frames.append(frame.copy())
            return frame
        
        if self.target_roi is None:
            if len(self.previous_frames) >= self.frames_to_keep:
                self.previous_frames.pop(0)
            self.previous_frames.append(frame.copy())
            return frame
        
        target_x, target_y, target_w, target_h = self.target_roi
        if target_w <= 0 or target_h <= 0:
            return frame
        
        target_region = frame[target_y:target_y + target_h, target_x:target_x + target_w]
        
        if len(self.previous_frames) >= self.frames_to_keep:
            self.previous_frames.pop(0)
        self.previous_frames.append(frame.copy())
        
        if len(self.previous_frames) < 2:
            return frame
        
        pre_impact_frame = self.previous_frames[0]
        pre_impact_region = pre_impact_frame[target_y:target_y + target_h, target_x:target_x + target_w]
        
        if pre_impact_region.size == 0 or target_region.size == 0:
            return frame
        
        hsv_current = cv2.cvtColor(target_region, cv2.COLOR_BGR2HSV)
        hsv_pre = cv2.cvtColor(pre_impact_region, cv2.COLOR_BGR2HSV)
        
        h_diff = cv2.absdiff(hsv_current[:, :, 0], hsv_pre[:, :, 0])
        h_diff = (h_diff / 180 * 255).astype(np.uint8)
        s_diff = cv2.absdiff(hsv_current[:, :, 1], hsv_pre[:, :, 1])
        v_diff = cv2.absdiff(hsv_current[:, :, 2], hsv_pre[:, :, 2])
        
        _, v_thresh = cv2.threshold(v_diff, self.brightness_threshold, 255, cv2.THRESH_BINARY)
        _, h_thresh = cv2.threshold(h_diff, self.color_change_threshold, 255, cv2.THRESH_BINARY)
        _, s_thresh = cv2.threshold(s_diff, self.color_change_threshold, 255, cv2.THRESH_BINARY)
        
        combined_change = cv2.bitwise_or(cv2.bitwise_or(h_thresh, s_thresh), v_thresh)
        
        kernel_size = max(1, int(min(target_w, target_h) * 0.05))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        combined_change = cv2.morphologyEx(combined_change, cv2.MORPH_OPEN, kernel)
        combined_change = cv2.morphologyEx(combined_change, cv2.MORPH_CLOSE, kernel)
        
        white_pixels = cv2.countNonZero(combined_change)
        total_pixels = target_w * target_h
        change_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
        
        if change_ratio > self.explosion_change_ratio_threshold:
            self.explosion_detected = True
            self.key_frames["explosion"] = frame.copy()
            self.key_frames_time["explosion"] = self.frame_counter / self.fps
            
            x, y, w, h = self.target_roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "爆炸发生", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame
    
    def detect_explosion_end(self, frame):
        """严格检测爆炸结束 - 目标区域完全恢复稳定"""
        # 如果已经检测到爆炸结束，直接返回
        if self.explosion_end_detected:
            return frame
        # 如果未检测到爆炸，不检测爆炸结束
        if not self.explosion_detected:
            return frame
        # 提取目标区域进行分析
        target_x, target_y, target_w, target_h = self.target_roi
        target_region = frame[target_y:target_y + target_h, target_x:target_x + target_w]
        # 保存当前目标区域
        if len(self.previous_explosion_frames) >= self.explosion_frames_to_keep:
            self.previous_explosion_frames.pop(0)
        self.previous_explosion_frames.append(target_region.copy())
        # 如果没有足够的历史帧，返回
        if len(self.previous_explosion_frames) < self.explosion_frames_to_keep:
            return frame
            # 获取当前帧和前一帧进行对比
        curr_frame = self.previous_explosion_frames[-1]
        prev_frame = self.previous_explosion_frames[-2]
        # 计算多通道帧间差异
        frame_diff = cv2.absdiff(curr_frame, prev_frame)
        # 分别检查每个颜色通道的变化
        b_diff, g_diff, r_diff = cv2.split(frame_diff)
        # 对每个通道应用阈值处理
        _, b_thresh = cv2.threshold(b_diff, self.pixel_stability_threshold, 255, cv2.THRESH_BINARY)
        _, g_thresh = cv2.threshold(g_diff, self.pixel_stability_threshold, 255, cv2.THRESH_BINARY)
        _, r_thresh = cv2.threshold(r_diff, self.pixel_stability_threshold, 255, cv2.THRESH_BINARY)
        # 合并所有通道的变化
        combined_thresh = cv2.bitwise_or(cv2.bitwise_or(b_thresh, g_thresh), r_thresh)
        kernel_size = max(3, int(min(target_w, target_h) * 0.1))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)  # 增大核尺寸以更好地连接相邻区域
        dilated = cv2.dilate(combined_thresh, kernel, iterations=1)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        # 寻找连通区域
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 分析连通区域的大小
        significant_changes = False
        if target_w * target_h < 1000:
            min_contour_area = target_w * target_h * (self.min_contour_area_ratio + 0.1)  # 更小的阈值捕获更微小的变化
        else:
            min_contour_area = target_w * target_h * self.min_contour_area_ratio
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_contour_area:
                significant_changes = True
                # 在原图上标记变化区域（可用于调试）
                cv2.drawContours(frame[target_y:target_y + target_h, target_x:target_x + target_w],
                                 [contour], -1, (0, 0, 255), 1)
                break
        # 如果没有显著的成片变化区域，增加稳定计数
        if not significant_changes:
            self.stability_counter += 1
        else:
            self.stability_counter = 0  # 重置稳定计数器
        # 显示当前的稳定计数和阈值（可选，用于调试）
        cv2.putText(frame, f"Stability: {self.stability_counter}/{self.stability_threshold}",
                    (target_x, target_y - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # 如果连续多帧都没有显著变化，确认爆炸结束
        if self.stability_counter >= self.stability_threshold:
            self.explosion_end_detected = True
            self.key_frames["explosion_end"] = frame.copy()
            self.key_frames_time["explosion_end"] = self.frame_counter / self.fps
            # 在目标区域绘制绿色边框表示爆炸结束
            cv2.rectangle(frame, (target_x, target_y),
                          (target_x + target_w, target_y + target_h),
                          (0, 255, 0), 2)
            # 添加爆炸结束标记
            text_pos = (target_x, target_y - 70)
            frame = self.put_chinese_text(frame, "爆炸结束!", text_pos, 30, (0, 255, 0))
        return frame
    
    def format_time(self, seconds):
        if seconds is None or not isinstance(seconds, (int, float)):
            return "未检测到"
        
        if seconds < 0:
            return "无效时间"
        
        total_seconds = int(seconds)
        fractional = seconds - total_seconds
        ms = int(round(fractional * 1000))
        
        if ms >= 1000:
            ms -= 1000
            total_seconds += 1
        
        days, remaining = divmod(total_seconds, 86400)
        hours, remaining = divmod(remaining, 3600)
        minutes, seconds = divmod(remaining, 60)
        
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"
        
        if days > 0:
            return f"{days}天 {time_str}"
        return time_str
    
    def put_chinese_text(self, img, text, position, font_size, color):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        try:
            if self.font_path:
                font = ImageFont.truetype(self.font_path, font_size)
            else:
                font = ImageFont.load_default()
            
            draw.text(position, text, font=font, fill=color[::-1])
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except:
            return img
    
    def process_video_without_display(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_counter += 1
            
            fg_mask = self.bg_subtractor.apply(frame)
            frame = self.detect_missile(frame, fg_mask)
            frame = self.detect_impact(frame)
            frame = self.detect_explosion(frame)
            frame = self.detect_explosion_end(frame)
            
            if (self.key_frames["missile_appearance"] is not None and
                self.key_frames["missile_impact"] is not None and
                self.key_frames["explosion"] is not None):
                pass
        
        self.cap.release()
        return self.key_frames, self.key_frames_time

class GetkeyImage:
    def __init__(self, capture_path="D:/tmp"):
        self.capture_path = capture_path
        if not os.path.exists(capture_path):
            os.makedirs(capture_path)
        
    def call(self, video_path, target_points=None):
        analyzer = MissileStrikeAnalyzer(video_path)
        
        if target_points is not None:
            (x1, y1), (x2, y2) = target_points
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            target_roi = (x, y, w, h)
            analyzer.target_roi = target_roi
        else:
            print("正在打开视频以选择目标区域...")
            analyzer.select_target_roi()
        
        key_frames, key_frames_time = analyzer.process_video_without_display()
        
        # 打开视频文件以重新读取原始帧（不带标记）
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("无法重新打开视频文件以获取原始关键帧")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        result = []
        event_names = {
            "missile_appearance": "导弹出现",
            "missile_impact": "导弹与目标区域接触",
            "explosion": "导弹爆炸",
            "explosion_end": "爆炸结束"
        }
        
        for event_key, frame in key_frames.items():
            if frame is not None and key_frames_time[event_key] is not None:
                # 计算关键帧在视频中的帧索引
                frame_index = int(key_frames_time[event_key] * fps)
                
                # 将视频定位到该帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                
                # 读取原始帧（不带任何标记）
                ret, original_frame = cap.read()
                if not ret or original_frame is None:
                    print(f"无法读取关键帧 {event_key} 对应的原始帧")
                    continue
                    
                filename = f"{os.path.basename(video_path)}.{event_key}.png"
                filepath = os.path.join(self.capture_path, filename)
                
                cv2.imwrite(filepath, original_frame)
                
                time_sec = key_frames_time[event_key]
                time_str = analyzer.format_time(time_sec) if time_sec is not None else "00:00:00.000"
                
                result.append({
                    "capture": filepath,
                    "time": time_str,
                    "eventName": event_names[event_key],
                    "desc": "",
                    "eventKey": event_key
                })
        
        cap.release()
        return result

class Image2Text:
    def __init__(self, api_key=None):
        self.api_key = api_key or "sk-2d64c5ef5b434265b535122a12aa9cea"
        self.scene_rules = {
            "missile_appearance": {
                "forbidden_words": ["爆炸", "火球", "烟雾", "冲击波", "碎片", "损毁", "火花", "接触", "击中"],
                "required_words": ["导弹"]
            },
            "missile_impact": {
                "forbidden_words": ["爆炸", "火球", "烟雾", "碎片", "损毁", "结束", "消失"],
                "required_words": ["接触"]
            },
            "explosion": {
                "forbidden_words": ["导弹", "飞行", "尾焰", "弹体", "接触", "击中", "出现", "目标区域"],
                "required_words": ["火球", "爆炸"]
            },
            "explosion_end": {
                "forbidden_words": ["导弹", "飞行", "尾焰", "弹体", "爆炸", "火球", "冲击波", "接触", "击中", "目标区域"],
                "required_words": ["烟雾"]
            }
        }
    
    def call(self, image_path, event_key, max_retries=3):
        """
        主要方法：通过API生成描述，并确保描述符合场景要求
        1. 首次调用使用标准prompt
        2. 如果描述不符合要求，构造更严格的prompt重新调用
        3. 最多重试max_retries次
        4. 重试失败后使用默认描述（最后手段）
        """
        # 获取基础prompt
        base_prompt = self.get_scene_prompt(event_key)
        
        # 尝试生成符合要求的描述
        for retry in range(max_retries + 1):
            # 生成当前尝试的prompt
            current_prompt = base_prompt if retry == 0 else self.get_retry_prompt(base_prompt, event_key, last_validation_result, retry)
            
            # 调用API获取描述
            description = self.analyze_image(image_path, current_prompt)
            
            # 验证描述
            is_valid, validation_result = self.validate_description(description, event_key)
            
            # 如果描述有效，返回（但确保不超过50字）
            if is_valid:
                return self.ensure_length_limit(description)
            
            # 保存验证结果用于下一次重试
            last_validation_result = validation_result
            
            # 短暂等待，避免API调用过于频繁
            time.sleep(0.5)
        
        # 如果多次重试都失败，返回一个安全的默认描述
        return self.get_safe_default_description(event_key)
    
    def get_scene_prompt(self, event_key):
        """
        为每个关键帧类型生成基础描述提示
        """
        prompts = {
            "missile_appearance": """这是一个导弹出现的关键帧，图像中导弹位于画面边缘（通常有绿色框标记），尚未接触目标区域。
请严格描述：
1. 导弹的外观特征（颜色、形状、大小）
2. 导弹的飞行方向和速度
3. 导弹与目标区域的相对位置
4. 是否有尾焰等特征
注意：此时没有爆炸、烟雾或冲击波！
请用50字以内的一句话描述，必须包含'导弹'一词，禁止出现'爆炸'、'烟雾'、'火球'等词汇。""",
            
            "missile_impact": """这是一个导弹与目标接触的关键帧，导弹正在接触目标区域（蓝色框标记）。
请严格描述：
1. 接触的瞬间状态
2. 接触点位置
3. 产生的火花或小型冲击波
4. 导弹与目标的角度
注意：此时有火花但无大规模爆炸！
请用50字以内的一句话描述，必须包含'接触'一词，禁止出现'爆炸'、'火球'、'烟雾'等词汇。""",
            
            "explosion": """这是一个爆炸发生的关键帧，图像显示爆炸初始阶段（红色框标记目标区域）。
请严格描述：
1. 爆炸火球的颜色和形状
2. 冲击波的扩散情况
3. 亮度和规模
4. 周围环境的初始影响
注意：此时导弹已不存在！
请用50字以内的一句话描述，必须包含'火球'或'爆炸'一词，禁止出现'导弹'、'飞行'、'尾焰'等词汇。""",
            
            "explosion_end": """这是一个爆炸结束的关键帧，图像显示爆炸后期（绿色框标记原目标区域）。
请严格描述：
1. 烟雾的颜色、浓度和扩散方向
2. 目标区域的损毁程度
3. 碎片分布情况
4. 是否有余烬
注意：此时爆炸已完成，无火球，导弹已不存在！
请用50字以内的一句话描述，必须包含'烟雾'一词，禁止出现'导弹'、'爆炸'、'火球'、'冲击波'等词汇。"""
        }
        return prompts.get(event_key, "请用50字以内描述画面内容。")
    
    def get_retry_prompt(self, base_prompt, event_key, validation_result, retry_count):
        """
        生成重试用的更严格prompt
        """
        reason = validation_result["reason"]
        
        if "包含禁止词汇" in reason:
            word = reason.split("'")[1]
            return f"第{retry_count}次重试：请重新描述，特别注意不能包含'{word}'这个词。{base_prompt}"
        
        elif "缺少必要词汇" in reason:
            required_words = "或".join(self.scene_rules[event_key]["required_words"])
            return f"第{retry_count}次重试：请重新描述，必须包含'{required_words}'一词。{base_prompt}"
        
        elif "长度超标" in reason:
            return f"第{retry_count}次重试：请重新描述，严格控制在50字以内。{base_prompt}"
        
        return f"第{retry_count}次重试：请严格遵守要求重新描述。{base_prompt}"
    
    def validate_description(self, description, event_key):
        """
        验证描述是否符合场景要求
        返回 (is_valid, validation_result)
        validation_result 包含验证详情
        """
        rules = self.scene_rules.get(event_key, {})
        validation_result = {"is_valid": True, "reason": ""}
        
        # 检查禁止词汇
        for word in rules.get("forbidden_words", []):
            if word in description:
                validation_result["is_valid"] = False
                validation_result["reason"] = f"包含禁止词汇: '{word}'"
                return False, validation_result
        
        # 检查必要词汇
        has_required_word = False
        for word in rules.get("required_words", []):
            if word in description:
                has_required_word = True
                break
        
        if not has_required_word and rules.get("required_words"):
            validation_result["is_valid"] = False
            validation_result["reason"] = f"缺少必要词汇: {rules['required_words']}"
            return False, validation_result
        
        # 检查长度
        if len(description) > 50:
            validation_result["is_valid"] = False
            validation_result["reason"] = f"长度超标: {len(description)}字 > 50字"
            return False, validation_result
        
        return True, validation_result
    
    def ensure_length_limit(self, description):
        """确保描述不超过50字（仅在最后返回时做此修正）"""
        if len(description) > 50:
            # 尝试找到最后一个标点符号进行截断
            last_punctuation = max(description.rfind('。'), description.rfind('，'), description.rfind('！'), description.rfind('？'))
            if last_punctuation > 0 and last_punctuation < 47:
                return description[:last_punctuation + 1]
            return description[:47] + "..."
        return description
    
    def get_safe_default_description(self, event_key):
        """仅在多次调用API生成描述失败后使用的安全默认描述"""
        defaults = {
            "missile_appearance": "一枚细长导弹从画面边缘飞入，朝目标方向飞行，与目标区域尚有距离",
            "missile_impact": "导弹尖端与目标区域中心点接触，有尘土飞扬，产生小型冲击波",
            "explosion": "有明亮橙红色火球出现并迅速扩散，冲击波席卷周围，亮度极高",
            "explosion_end": "灰色烟雾基本散去，目标区域严重损毁，爆炸基本结束"
        }
        return defaults.get(event_key, "画面显示关键事件")
    
    def analyze_image(self, image_path, prompt):
        """调用API获取图像描述"""
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "qwen-vl-plus",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": f"data:image/jpeg;base64,{image_base64}"
                            },
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            }
        }
        
        try:
            response = requests.post(url, json=data, headers=headers, timeout=30)
            if response.status_code == 200:
                result = response.json()
                content = result["output"]["choices"][0]["message"]["content"]
                
                if isinstance(content, list) and len(content) > 0:
                    if isinstance(content[0], dict) and 'text' in content[0]:
                        return content[0]['text']
                    else:
                        return str(content[0])
                else:
                    return str(content)
            else:
                return f"API错误 {response.status_code}"
        except Exception as e:
            return f"请求失败: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='导弹视频关键帧提取与描述系统')
    parser.add_argument('video_path', type=str, help='视频文件路径')
    parser.add_argument('coords', type=int, nargs='*', help='目标区域坐标 (x1 y1 x2 y2)')
    parser.add_argument('--capture_path', type=str, default='D:/tmp', help='关键帧保存路径')
    
    args = parser.parse_args()
    
    target_points = None
    if len(args.coords) == 4:
        x1, y1, x2, y2 = args.coords
        target_points = [(x1, y1), (x2, y2)]
        print(f"使用提供的目标区域坐标: ({x1}, {y1}), ({x2}, {y2})")
    elif len(args.coords) > 0:
        print(f"警告：需要提供4个坐标值 (x1 y1 x2 y2)，但只提供了{len(args.coords)}个值，将使用交互式选择")
    
    anys = GetkeyImage(capture_path=args.capture_path)
    res = anys.call(args.video_path, target_points=target_points)
    
    for cap in res:
        i2t = Image2Text()
        # 传递eventKey参数用于精确描述，最多重试3次
        cap['desc'] = i2t.call(cap["capture"], cap["eventKey"], max_retries=3)
        print(f"已生成 {cap['eventName']} 的描述: {cap['desc']}")
    
    print("\n关键帧信息:")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    
    print("\n结果已保存到 result.json")

if __name__ == '__main__':
    main()