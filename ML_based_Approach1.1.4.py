import cv2          # OpenCV库，用于计算机视觉处理
import numpy as np  # 数值计算库
import argparse     # 命令行参数解析
import os           # 操作系统接口
from datetime import timedelta  # 时间处理
from PIL import Image, ImageDraw, ImageFont  # 图像处理和字体绘制
import tkinter as tk # 导入 tkinter 用于获取屏幕尺寸

class MissileStrikeAnalyzer:    #导弹打击分析主类（重要参数:video_path）
    def __init__(self, video_path):     #初始化视频参数和检测器（关键技术：视频捕获、背景建模；重要参数：history=500, varThreshold=16）
        # 视频捕获对象初始化
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("无法打开视频文件")
        # 获取视频参数：帧率、尺寸、总帧数
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 用于显示结果的画布
        self.display_width = self.frame_width * 2
        self.display_height = self.frame_height
        # 初始化关键帧
        self.key_frames = {  # 四个关键事件对应的帧
            "missile_appearance": None, # 导弹出现
            "missile_impact": None,     # 导弹接触目标
            "explosion": None,          # 爆炸发生
            "explosion_end": None  # 新增爆炸结束关键帧
        }
        self.key_frames_time = {
            "missile_appearance": None,
            "missile_impact": None,
            "explosion": None,
            "explosion_end": None  # 新增爆炸结束时间戳
        }
        # 目标区域
        self.target_roi = None
        # 背景减除器
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16,
                                                                detectShadows=False)
        # 光流参数
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # 上一帧的灰度图和特征点
        self.prev_gray = None
        self.prev_points = None
        # 导弹轨迹和边界框
        self.missile_trajectory = []
        self.missile_bbox = None
        # 爆炸检测参数
        self.explosion_threshold = 200  # 亮度变化阈值
        self.explosion_min_area = self.frame_width * self.frame_height * 0.01  # 最小爆炸面积
        # 状态标志
        self.missile_detected = False
        self.impact_detected = False
        self.explosion_detected = False
        self.explosion_end_detected = False  # 新增：爆炸结束标志
        # 帧计数器
        self.frame_counter = 0
        # 碰撞后导弹消失的计时器
        self.impact_frame_count = 0
        self.cooldown_frames = int(self.fps * 0.02)  # 碰撞后 0.02秒内不检测导弹
        # 以前的帧，用于爆炸检测
        self.previous_frames = []
        self.frames_to_keep = 5  # 保存多少帧用于爆炸检测
        # 爆炸亮度和颜色变化阈值
        self.brightness_threshold = 30
        self.color_change_threshold = 40
        self.explosion_change_ratio_threshold = 0.1  # （确保主类也定义，防止引用异常）
        self.explosion_change_ratio_threshold = 0.1
        self.min_roi_pixels = 1024  # 32*32
        # 爆炸结束检测参数
        self.stability_threshold = 15  # 连续稳定帧数阈值，增加到15帧以确保真正稳定
        self.stability_counter = 0  # 稳定帧计数器
        self.pixel_stability_threshold = 5  # 像素稳定性阈值，降低到5以捕获微小变化
        self.previous_explosion_frames = []  # 存储爆炸后的几帧用于分析稳定性
        self.explosion_frames_to_keep = 10  # 增加到10帧以进行更可靠的稳定性分析
        self.min_contour_area_ratio = 0.02  # 显著变化区域的最小面积比例，降低到目标区域的0.2%

        # 中文字体
        # 尝试加载系统中的中文字体，根据实际情况可能需要修改
        try:
            # Windows 系统
            self.font_path = "C:/Windows/Fonts/simhei.ttf"
            if not os.path.exists(self.font_path):
                # Linux 系统
                self.font_path = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
                if not os.path.exists(self.font_path):
                    # macOS 系统
                    self.font_path = "/System/Library/Fonts/PingFang.ttc"
                    if not os.path.exists(self.font_path):
                        # 如果找不到系统中文字体，使用默认字体
                        self.font_path = None
        except:
            self.font_path = None

    def select_target_roi(self):
        """选择目标区域，窗口保持原视频高宽比但缩小尺寸"""
        ret, first_frame = self.cap.read()
        if not ret:
            raise ValueError("无法读取视频的第一帧")
        
        # 计算缩小后的窗口尺寸（保持原视频高宽比）
        scale_factor = 0.6  # 缩小为原尺寸的一半
        display_width = int(self.frame_width * scale_factor)
        display_height = int(self.frame_height * scale_factor)
        
        # 调整窗口位置居中
        screen_width, screen_height = 1920, 1080  # 默认屏幕尺寸，实际会通过tkinter获取
        try:
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
        except:
            pass
        
        # 计算窗口位置使其居中
        x_pos = max(0, (screen_width - display_width) // 2)
        y_pos = max(0, (screen_height - display_height) // 2)
        
        # 创建可调整大小的窗口
        cv2.namedWindow("Select Target Roi", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select Target Roi", display_width, display_height)
        cv2.moveWindow("Select Target Roi", x_pos, y_pos)
        
        # 添加中文提示
        frame_with_text = self.put_chinese_text(first_frame, "请选择目标区域，按回车确认", (20, 40), 36, (0, 255, 255))
        
        # 显示缩小后的窗口
        cv2.imshow("Select Target Roi", frame_with_text)
        self.target_roi = cv2.selectROI("Select Target Roi", frame_with_text, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Target Roi")
        
        # 重置视频
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def check_rectangles_overlap(self, rect1, rect2):   #辅助方法：检查两个矩形是否重叠（使用坐标比较法）
        """检查两个矩形是否重叠
        rect格式: (x, y, width, height)
        """
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        # 如果一个矩形在另一个的右侧
        if x1 >= x2 + w2 or x2 >= x1 + w1:
            return False
            # 如果一个矩形在另一个的下方
        if y1 >= y2 + h2 or y2 >= y1 + h1:
            return False
            # 否则重叠
        return True

    # --- 改进后的 detect_missile 函数 ---
    def detect_missile(self, frame, fg_mask):
        """改进版：检测并跟踪导弹，减少误检"""

        # 如果已经检测到碰撞，并且在冷却期间，不检测导弹
        if self.impact_detected and self.impact_frame_count < self.cooldown_frames:
            self.impact_frame_count += 1
            self.missile_bbox = None # 清除导弹边界框
            return frame

        # 如果已经检测到碰撞和爆炸，不再检测新的导弹
        if self.impact_detected and self.explosion_detected:
            self.missile_bbox = None # 爆炸后导弹消失
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 应用背景减除并进行形态学处理
        _, moving_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
        kernel_size = max(3, frame.shape[1] // 200)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        moving_mask = cv2.morphologyEx(moving_mask, cv2.MORPH_OPEN, kernel)
        moving_mask = cv2.morphologyEx(moving_mask, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(moving_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 收集可能的导弹候选
        candidates = []
        frame_area = self.frame_width * self.frame_height
        min_area_abs = 100 # 最小像素面积，防止检测微小噪点
        min_area_ratio = 0.0001 # 最小相对面积 (0.01%)
        max_area_ratio = 0.05   # 最大相对面积 (5%)
        min_aspect_ratio = 2.0  # 最小长宽比 (细长)

        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            # 过滤极小或无效区域
            if area < min_area_abs or w == 0 or h == 0:
                continue

            # 过滤过大或过小的区域 (相对面积)
            area_ratio = area / frame_area
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                 continue

            # 计算长宽比 (考虑水平或垂直)
            aspect_ratio = max(w, h) / min(w, h)

            # 过滤不符合细长特征的区域
            if aspect_ratio < min_aspect_ratio:
                 continue

            # 如果通过了基本过滤，添加到候选列表
            candidates.append({'bbox': (x, y, w, h), 'area': area, 'aspect_ratio': aspect_ratio, 'center': (x + w // 2, y + h // 2)})

        # --- 改进点 1: 收集所有候选并基于历史信息选择最佳 ---
        best_candidate = None

        if candidates:
            if self.missile_bbox is not None:
                # 如果上一帧有导弹，寻找当前帧中最接近的候选
                prev_center = (self.missile_bbox[0] + self.missile_bbox[2] // 2, self.missile_bbox[1] + self.missile_bbox[3] // 2)
                min_distance = float('inf')

                for candidate in candidates:
                    dist = np.sqrt((candidate['center'][0] - prev_center[0])**2 + (candidate['center'][1] - prev_center[1])**2)
                    # 检查移动距离是否合理 (防止跳变)
                    # 假设最大像素速度为 frame_width/2 每秒
                    max_reasonable_dist_per_frame = (self.frame_width / 2) / self.fps if self.fps > 0 else self.frame_width / 10
                    if dist < min_distance and dist < max_reasonable_dist_per_frame * 1.5: # 允许1.5倍的合理速度波动
                        min_distance = dist
                        best_candidate = candidate

                # 如果找到了一个合理的接近的候选，就认为是导弹
                if best_candidate:
                    self.missile_bbox = best_candidate['bbox']
                    self.missile_trajectory.append(best_candidate['center'])
                    self.missile_detected = True # 确认导弹已检测到

            # 如果上一帧没有导弹，或者没有找到接近的候选，从当前候选列表中选择一个最可能的
            if best_candidate is None:
                 # 选择面积最大的候选作为首次检测或重新检测的目标
                 best_candidate = max(candidates, key=lambda c: c['area'])
                 self.missile_bbox = best_candidate['bbox']
                 self.missile_trajectory.append(best_candidate['center'])
                 # 首次检测到导弹，保存关键帧
                 if not self.missile_detected:
                     self.missile_detected = True
                     self.key_frames["missile_appearance"] = frame.copy()
                     self.key_frames_time["missile_appearance"] = self.frame_counter / self.fps

        else:
            # 如果没有找到任何候选，清除导弹状态
            self.missile_bbox = None
            # 注意：这里不重置 self.missile_detected = False，因为导弹可能只是暂时被遮挡
            # 如果需要更严格的丢失跟踪判断，可以在这里添加逻辑 (例如连续N帧未检测到则重置)

        # --- 改进点 1 结束 ---

        # --- 改进点 2: 仅绘制选定的最佳候选的边界框 ---
        if self.missile_bbox is not None:
            x, y, w, h = self.missile_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 绘制轨迹点 (可选)
            # for point in self.missile_trajectory:
            #     cv2.circle(frame, point, 2, (0, 0, 255), -1)
            cv2.putText(frame, "Missile", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # --- 改进点 2 结束 ---


        # --- 改进点 3: 光流跟踪逻辑调整 ---
        # 光流跟踪可以独立于轮廓检测进行，用于辅助平滑轨迹或验证运动
        # 可以在检测到导弹后，在导弹区域附近检测特征点进行更精确的跟踪
        # 但为了保持与原代码结构相似，我们继续在整个运动区域检测特征点
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            # 初始特征点检测 (在运动区域内)
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=moving_mask, **self.feature_params)
        else:
            # 如果有跟踪点，更新它们
            if self.prev_points is not None and len(self.prev_points) > 0:
                next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, self.prev_points, None, **self.lk_params)

                if next_points is not None:
                    # 选择好的点
                    good_next = next_points[status == 1]
                    good_prev = self.prev_points[status == 1]

                    # 绘制跟踪线 (可选，可能影响性能)
                    # for i, (new, old) in enumerate(zip(good_next, good_prev)):
                    #     a, b = new.ravel()
                    #     c, d = old.ravel()
                    #     frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 1)

                    # 更新跟踪点，如果点太少则重新检测
                    if len(good_next) >= 5: # 至少保留5个点
                        self.prev_points = good_next.reshape(-1, 1, 2)
                    else:
                        # 点丢失过多，重新检测特征点 (在运动区域内)
                        self.prev_points = cv2.goodFeaturesToTrack(gray, mask=moving_mask, **self.feature_params)
                else:
                     # 如果next_points是None，说明跟踪失败，重新检测特征点
                     self.prev_points = cv2.goodFeaturesToTrack(gray, mask=moving_mask, **self.feature_params)
            else:
                # 如果没有跟踪点，尝试检测新的点 (在运动区域内)
                self.prev_points = cv2.goodFeaturesToTrack(gray, mask=moving_mask, **self.feature_params)

        self.prev_gray = gray.copy()
        # --- 改进点 3 结束 ---

        return frame

    # --- 其他函数保持不变 ---

    def detect_impact(self, frame): #检测导弹与目标的接触（关键技术：区域重叠检测；使用check_rectangles_overlap方法）
        
        """检测导弹与目标的接触"""
        if self.missile_detected and not self.impact_detected and self.missile_bbox is not None:
            
            # 检查导弹边界框是否与目标区域重叠
            try:
                overlap = self.check_rectangles_overlap(self.missile_bbox, self.target_roi)
            except ValueError as e:
                print(f"Overlap检测错误: {e}")
                return frame
            if overlap:
                
                self.impact_detected = True
                self.impact_frame_count = 0  # 重置碰撞后计数器
                self.key_frames["missile_impact"] = frame.copy()
                self.key_frames_time["missile_impact"] = self.frame_counter / self.fps
                
                # 在碰撞位置绘制黄色碰撞标记
                missile_x, missile_y, missile_w, missile_h = self.missile_bbox
                target_x, target_y, target_w, target_h = self.target_roi
                # 计算重叠区域
                overlap_x = max(missile_x, target_x)
                overlap_y = max(missile_y, target_y)
                overlap_w = min(missile_x + missile_w, target_x + target_w) - overlap_x
                overlap_h = min(missile_y + missile_h, target_y + target_h) - overlap_y
                
                # 在重叠区域绘制黄色标记
                cv2.rectangle(frame, (overlap_x, overlap_y),
                              (overlap_x + overlap_w, overlap_y + overlap_h),
                              (0, 255, 255), -1)  # 填充重叠区域
                # 在原始边界框上方添加碰撞文字
                text_pos = (target_x, target_y - 10)
                frame = self.put_chinese_text(frame, "碰撞!", text_pos, 30, (0, 0, 255))

        # 绘制目标区域
        if self.target_roi:
            x, y, w, h = self.target_roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # 绘制导弹区域 - 只在未检测到碰撞时显示
        if self.missile_bbox and not self.impact_detected:
            x, y, w, h = self.missile_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame

    def detect_explosion(self, frame):  #检测爆炸和烟尘发生（关键技术：HSV颜色空间分析；重要参数：brightness_threshold=30）
        """检测爆炸和烟尘"""
        # 如果已经检测到爆炸，直接返回当前帧，避免重复计算
        if self.explosion_detected:
            return frame

        # --- 改动点 A 开始 ---
        # 如果未检测到导弹，不检测爆炸 (根据需求: 事件3依赖事件1)
        # 原始代码是 if not self.impact_detected: return frame (事件3依赖事件2)
        if not self.missile_detected:
             # 保存当前帧用于后续爆炸检测 (即使未检测到导弹，也可以开始缓存，等待导弹出现)
             if len(self.previous_frames) >= self.frames_to_keep:
                 self.previous_frames.pop(0)
             self.previous_frames.append(frame.copy())
             return frame # 导弹未出现，跳过爆炸检测逻辑
        # --- 改动点 A 结束 ---

        # 如果目标区域未选择，无法检测爆炸
        if self.target_roi is None:
            # 如果未检测到导弹，不检测爆炸 (根据需求: 事件3依赖事件1)
            # 在此分支下也需要缓存帧，以便导弹出现后立即进行对比
            if len(self.previous_frames) >= self.frames_to_keep:
                self.previous_frames.pop(0)
            self.previous_frames.append(frame.copy())
            return frame # 没有目标区域，无法检测爆炸


        # 提取目标区域进行分析
        target_x, target_y, target_w, target_h = self.target_roi
        # 检查目标区域尺寸是否有效
        if target_w <= 0 or target_h <= 0:
             return frame # 目标区域无效

        target_region = frame[target_y:target_y + target_h, target_x:target_x + target_w]

        # 如果没有足够的历史帧，返回
        # 在进行爆炸检测前缓存当前帧
        if len(self.previous_frames) >= self.frames_to_keep:
            self.previous_frames.pop(0)
        self.previous_frames.append(frame.copy())

        # 如果没有足够的历史帧来计算差异，返回
        if len(self.previous_frames) < 2:
            return frame
        
        # 获取当前帧和碰撞前的帧进行对比
        # NOTE: 现在爆炸检测不依赖碰撞，所以对比帧应该是当前帧和缓存中的某个“相对稳定”帧
        # 原始代码这里是 self.previous_frames[0] (第一个缓存帧)，这可以作为一个简单的“碰撞前”帧替代
        # 或者更复杂的，选择导弹出现前的一帧作为基准帧
        # 为了最小化改动并符合“出现导弹后检测爆炸”的要求，我们使用缓存中的第一帧作为对比基准
        pre_impact_frame = self.previous_frames[0]
        pre_impact_region = pre_impact_frame[target_y:target_y + target_h, target_x:target_x + target_w]

        # 检查提取的区域是否有效
        if pre_impact_region.size == 0 or target_region.size == 0:
             return frame


        # 转换为 HSV 色彩空间以分析颜色变化
        hsv_current = cv2.cvtColor(target_region, cv2.COLOR_BGR2HSV)
        hsv_pre = cv2.cvtColor(pre_impact_region, cv2.COLOR_BGR2HSV)
        # 计算亮度和颜色差异
        h_diff = cv2.absdiff(hsv_current[:, :, 0], hsv_pre[:, :, 0])
        h_diff = (h_diff / 180 * 255).astype(np.uint8)
        s_diff = cv2.absdiff(hsv_current[:, :, 1], hsv_pre[:, :, 1])
        v_diff = cv2.absdiff(hsv_current[:, :, 2], hsv_pre[:, :, 2])
        
        # 检测大幅度亮度变化（爆炸通常伴随亮度急剧增加）
        _, v_thresh = cv2.threshold(v_diff, self.brightness_threshold, 255, cv2.THRESH_BINARY)
        # 检测火焰和烟雾的颜色变化
        _, h_thresh = cv2.threshold(h_diff, self.color_change_threshold, 255, cv2.THRESH_BINARY)
        _, s_thresh = cv2.threshold(s_diff, self.color_change_threshold, 255, cv2.THRESH_BINARY)
        # 合并各种变化
        combined_change = cv2.bitwise_or(cv2.bitwise_or(h_thresh, s_thresh), v_thresh)
        # 应用形态学操作
        kernel_size = max(1, int(min(target_w, target_h) * 0.05))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        combined_change = cv2.morphologyEx(combined_change, cv2.MORPH_OPEN, kernel)
        combined_change = cv2.morphologyEx(combined_change, cv2.MORPH_CLOSE, kernel)
        
        # 计算变化区域（小区域增大阈值）
        white_pixels = cv2.countNonZero(combined_change)
        total_pixels = target_w * target_h
        # 防止除以零
        if total_pixels == 0:
            change_ratio = 0
        else:
            change_ratio = white_pixels / total_pixels
        base_threshold = self.explosion_change_ratio_threshold
        if total_pixels < self.min_roi_pixels and total_pixels > 0: # 添加 total_pixels > 0 检查
            scale_factor = (self.min_roi_pixels - total_pixels) / self.min_roi_pixels
            # 调整阈值计算方式，确保不会负数
            base_threshold = min(1.0, self.explosion_change_ratio_threshold + scale_factor * 0.15) # 防止阈值超过1.0

        # 如果变化比例超过阈值，认为爆炸发生
        if change_ratio > base_threshold:  # 调整这个阈值
            self.explosion_detected = True
            # 复制当前帧，而不是修改当前正在显示的帧
            self.key_frames["explosion"] = frame.copy()
            self.key_frames_time["explosion"] = self.frame_counter / self.fps
            # 在目标区域绘制红色边框表示爆炸
            cv2.rectangle(frame, (target_x, target_y),
                          (target_x + target_w, target_y + target_h),
                          (0, 0, 255), 2)
            # 添加爆炸标记
            text_pos = (target_x, target_y - 40)
            frame = self.put_chinese_text(frame, "爆炸!", text_pos, 30, (0, 0, 255))

        return frame

    def detect_explosion_end(self, frame):  #检测爆炸结束（关键技术：多帧稳定性分析；重要参数：stability_threshold=15,需要连续15帧稳定才确认爆炸结束）
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

    def format_time(self, seconds):     #辅助方法：时间格式化
        """将秒数格式化为[天] HH:MM:SS.ms 格式"""
        # 输入有效性检查
        if seconds is None or not isinstance(seconds, (int, float)):
            # isinstance(seconds, (int, float)) 检查：
            # 验证seconds参数是否是整数或浮点数类型
            # 避免传入字符串、对象等非法类型导致计算错误
            return "未检测到"
        # 处理负数输入（根据实际需求可选）
        if seconds < 0:
            return "无效时间"
        
        # 分离整数秒和小数部分
        total_seconds = int(seconds)          # 取整数部分的总秒数
        fractional = seconds - total_seconds   # 获取小数部分（0.xxx秒）
        
        # 毫秒四舍五入处理
        ms = int(round(fractional * 1000))     # 将0.xxx秒转为毫秒并四舍五入
        # 为什么要四舍五入？
        # 示例：0.1234秒=123.4毫秒 → 123毫秒（直接取整会损失精度）
        # 示例：0.9995秒=999.5毫秒 → 四舍五入为1000毫秒（需要进位处理）
        
        # 处理毫秒进位（当四舍五入后满1000ms时）
        if ms >= 1000:
            ms -= 1000            # 减去1000毫秒（保留0-999范围）
            total_seconds += 1    # 总秒数增加1秒
        
        # 使用divmod分解时间分量（自动处理进位）
        days, remaining = divmod(total_seconds, 86400)  
        # divmod(a, b)返回(商,余数)
        # 86400秒=24小时，得到天数（days）和剩余秒数（remaining）
        
        hours, remaining = divmod(remaining, 3600)  
        # 3600秒=1小时，得到小时数（hours）和剩余秒数（remaining）
        
        minutes, seconds = divmod(remaining, 60)  
        # 60秒=1分钟，得到分钟数（minutes）和剩余秒数（seconds）
        
        # 格式化时间字符串
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"
        # :02d表示用两位显示，不足补零（如5秒→05）
        # :03d表示用三位显示毫秒，不足补零（如5毫秒→005）
        
        # 动态添加天数显示
        #if days > 0:
        return f"{days}天 {time_str}"  # 当有天数时显示"X天"前缀
        #return time_str                     # 无天数时直接显示时分秒

    def put_chinese_text(self, img, text, position, font_size, color):  #辅助方法：在图片上添加中文文本（关键技术：PIL字体渲染）
        """在图片上添加中文文本"""
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # 初始化PIL的绘图接口，用于在图像上绘制文本或图形
        draw = ImageDraw.Draw(img_pil)
        
        # 尝试加载中文字体
        try:
            if self.font_path:
                font = ImageFont.truetype(self.font_path, font_size)
            else:
                # 如果没有找到中文字体，使用默认字体
                font = ImageFont.load_default()
            # 在图片上绘制文本
            draw.text(position, text, font=font, fill=color[::-1])  # RGB->BGR
            # 将PIL的RGB图像转回OpenCV的BGR格式
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        # 捕获所有异常，防止程序崩溃，返回原始图像
        except Exception as e:
            print(f"添加中文文本时出错: {e}")
            # 发生错误时，返回原始图像
            return img

    def resize_with_aspect_ratio(self, image, target_width, target_height, bg_color=(0, 0, 0)):
        """
        调整图像大小以适应目标尺寸，同时保持宽高比。
        如有必要，使用背景色进行填充。
        (代码省略，请参考之前回答中的实现)
        """
        if image is None: return np.full((target_height, target_width, 3), bg_color, dtype=np.uint8)
        h_orig, w_orig = image.shape[:2]
        if h_orig == 0 or w_orig == 0: return np.full((target_height, target_width, 3), bg_color, dtype=np.uint8)
        scale = min(target_width / w_orig, target_height / h_orig)
        new_w = max(1, int(w_orig * scale))
        new_h = max(1, int(h_orig * scale))
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        background = np.full((target_height, target_width, 3), bg_color, dtype=np.uint8)
        paste_x = (target_width - new_w) // 2
        paste_y = (target_height - new_h) // 2
        try:
            background[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = resized_image
        except ValueError as e:
            print(f"粘贴错误: {e}") # 添加错误打印
            return np.full((target_height, target_width, 3), bg_color, dtype=np.uint8) # 出错时返回背景
        return background
        # --- resize_with_aspect_ratio 结束 ---

    def create_display_frame(self, current_frame):  # 生成双屏显示画面（严格适应屏幕宽度）
        """创建显示帧，左右屏各占屏幕一半，内容保持宽高比缩放"""

        # --- 改动点 1: 获取屏幕尺寸并计算最终显示尺寸 ---
        # 创建一个临时的 tkinter 根窗口 (不显示)
        root = tk.Tk()
        # 获取主屏幕的宽度 (像素)
        screen_width = root.winfo_screenwidth()
        # 获取主屏幕的高度 (像素)
        screen_height = root.winfo_screenheight()
        # 销毁临时窗口
        root.destroy()

        # 目标最终显示宽度等于屏幕宽度
        final_display_width = screen_width
        # 目标最终显示高度：按比例缩放，保持原布局宽高比 (宽/高 = 2*fw/fh)，且不超过屏幕高度
        # 计算原始布局的宽高比
        original_layout_ratio = (self.frame_width * 2) / self.frame_height if self.frame_height > 0 else 16/9 # 防止除零
        # 根据目标宽度计算目标高度
        target_height_based_on_width = int(final_display_width / original_layout_ratio)

        # 最终高度不能超过屏幕高度
        final_display_height = min(target_height_based_on_width, screen_height)
        # 如果因为高度限制导致宽度也需要调整 (即 target_height_based_on_width > screen_height)
        if target_height_based_on_width > screen_height:
            # 重新根据最终高度计算最终宽度，保持比例
            final_display_width = int(final_display_height * original_layout_ratio)

        # --- 改动点 1 结束 ---

        # --- 改动点 2: 计算基于最终显示尺寸的面板和格子尺寸 ---
        # 左侧面板宽度为最终显示宽度的一半
        final_left_panel_width = final_display_width // 2
        # 右侧面板宽度为剩余部分 (处理奇数宽度)
        final_right_panel_width = final_display_width - final_left_panel_width
        # 右侧 2x2 网格中每个格子的宽度
        final_quadrant_width = final_right_panel_width // 2
        # 右侧 2x2 网格中每个格子的高度
        final_quadrant_height = final_display_height // 2
        # --- 改动点 2 结束 ---

        # --- 改动点 3: 使用最终显示尺寸创建画布 ---
        # 创建最终显示尺寸的空白画布
        display_frame = np.zeros((final_display_height, final_display_width, 3), dtype=np.uint8)
        # --- 改动点 3 结束 ---

        # --- 左侧面板：缩放并放置当前帧 ---
        # --- 改动点 4: 使用 resize_with_aspect_ratio 缩放当前帧以适应左面板 ---
        if current_frame is not None:
            # 将 *原始* current_frame 缩放以适应左面板 (final_left_panel_width, final_display_height)
            left_panel_content = self.resize_with_aspect_ratio(current_frame,
                                                               final_left_panel_width,
                                                               final_display_height)
            # 将缩放后的内容放置到画布的左侧
            display_frame[0:final_display_height, 0:final_left_panel_width] = left_panel_content
        else:
            # 如果帧无效，绘制黑色背景和文字
            cv2.rectangle(display_frame, (0, 0), (final_left_panel_width, final_display_height), (0, 0, 0), -1)
            # 文字位置基于最终左面板尺寸居中
            display_frame = self.put_chinese_text(display_frame, "无视频帧", (final_left_panel_width//2 - 50, final_display_height//2), 30, (255, 255, 255))
        # --- 改动点 4 结束 ---

        # --- 添加标题标签 ---
        # --- 改动点 5: 右侧标题 X 坐标基于最终左面板宽度 ---
        display_frame = self.put_chinese_text(display_frame, "当前视频", (10, 30), 36, (255, 255, 255))
        # 右侧标题从最终左面板宽度之后开始
        display_frame = self.put_chinese_text(display_frame, "关键事件", (final_left_panel_width + 10, 30), 36, (255, 255, 255))
        # --- 改动点 5 结束 ---

        # --- 显示状态信息 (位置不变，但画布尺寸已变) ---
        status_y = 70
        missile_status = "已检测" if self.missile_detected else "未检测"
        impact_status = "已检测" if self.impact_detected else "未检测"
        explosion_status = "已检测" if self.explosion_detected else "未检测"
        explosion_end_status = "已检测" if self.explosion_end_detected else "未检测"
        # 字体大小固定，视觉效果会随画布缩放而变化
        display_frame = self.put_chinese_text(
            display_frame,
            f"导弹: {missile_status} | 碰撞: {impact_status} | 爆炸: {explosion_status} | 结束: {explosion_end_status}",
            (10, status_y), 24, (255, 255, 0)
        )
        # --- 状态信息结束 ---

        # --- 右侧面板：2x2 关键帧网格 ---
        key_frame_events = [
            ("missile_appearance", "1. 导弹出现"),
            ("missile_impact", "2. 导弹接触目标"),
            ("explosion", "3. 爆炸发生"),
            ("explosion_end", "4. 爆炸结束")
        ]

        # 遍历关键事件并放置到对应格子
        for i, (event_key, event_label) in enumerate(key_frame_events):
            key_frame_image = self.key_frames.get(event_key)
            key_frame_time = self.key_frames_time.get(event_key)

            # --- 改动点 6: 使用最终尺寸计算格子坐标 ---
            # 计算行号和列号
            row = i // 2
            col = i % 2
            # 计算当前格子在 *最终画布* 中的起始 X 坐标
            quadrant_x_start = final_left_panel_width + col * final_quadrant_width
            # 计算当前格子在 *最终画布* 中的起始 Y 坐标
            quadrant_y_start = row * final_quadrant_height
            # 计算当前格子在 *最终画布* 中的结束 X 坐标
            quadrant_x_end = quadrant_x_start + final_quadrant_width
            # 计算当前格子在 *最终画布* 中的结束 Y 坐标
            quadrant_y_end = quadrant_y_start + final_quadrant_height
            # --- 改动点 6 结束 ---

            # --- 改动点 7: 使用 resize_with_aspect_ratio 缩放关键帧以适应格子 ---
            # 将 *原始* key_frame_image 缩放以适应格子 (final_quadrant_width, final_quadrant_height)
            quadrant_content = self.resize_with_aspect_ratio(key_frame_image,
                                                             final_quadrant_width,
                                                             final_quadrant_height)
            # 将缩放后的内容（带背景）放置到画布上对应的格子位置
            display_frame[quadrant_y_start:quadrant_y_end, quadrant_x_start:quadrant_x_end] = quadrant_content
            # --- 改动点 7 结束 ---

            # --- 改动点 8: 文本位置基于最终格子坐标 ---
            # 在格子内添加时间戳或占位符文本
            if key_frame_image is not None:
                time_str = self.format_time(key_frame_time)
                text_to_show = f"{event_label} - {time_str}"
                text_color = (0, 255, 0) # 绿色
            else:
                text_to_show = f"{event_label} - 等待检测"
                text_color = (128, 128, 128) # 灰色

            # 计算文本位置 (靠近格子底部)
            text_x = quadrant_x_start + 10
            text_y = quadrant_y_end - 30
            # 字体大小固定
            display_frame = self.put_chinese_text(display_frame, text_to_show, (text_x, text_y), 24, text_color)
            # --- 改动点 8 结束 ---

        # --- 右侧面板结束 ---

        # --- 添加当前时间 ---
        # --- 改动点 9: 文本位置基于最终显示高度 ---
        current_time_sec = self.frame_counter / self.fps if self.fps > 0 else 0
        current_time_str = self.format_time(current_time_sec)
        # Y 坐标基于最终显示高度
        display_frame = self.put_chinese_text(
            display_frame, f"当前时间: {current_time_str}",
            (10, final_display_height - 20), 24, (255, 255, 255)
        )
        # --- 改动点 9 结束 ---

        # --- 改动点 10: 移除所有末尾缩放逻辑 ---
        # 画布本身就是最终要显示的尺寸，无需再缩放
        # --- 改动点 10 结束 ---

        # --- 可选：更新实例变量 ---
        # 更新 self.display_width/height 以反映当前实际显示的画布尺寸
        self.display_width = final_display_width
        self.display_height = final_display_height
        # --- 可选更新结束 ---

        # 返回最终构建好的、尺寸严格适应屏幕的显示帧
        return display_frame
    def process_video(self, output_path=None):  #处理视频主循环（关键技术：帧处理链式调用）
        """处理视频并识别关键事件"""
        # 让用户选择目标区域，用于后续碰撞检测
        self.select_target_roi()
        
        # 准备视频写入器
        if output_path:
            # 在创建 VideoWriter 之前，先生成一帧显示帧来确定最终尺寸
            # 使用一个空白帧来避免依赖实际视频帧
            dummy_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            dummy_display_frame = self.create_display_frame(dummy_frame)
            final_output_height, final_output_width = dummy_display_frame.shape[:2]

            out = cv2.VideoWriter(output_path,
                                  cv2.VideoWriter_fourcc(*'XVID'),
                                  self.fps,
                                  (final_output_width, final_output_height))
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_counter += 1
            
            # 应用背景减除
            fg_mask = self.bg_subtractor.apply(frame)
            
            # 检测并跟踪导弹
            frame = self.detect_missile(frame, fg_mask)
            
            # 检测导弹与目标的接触
            frame = self.detect_impact(frame)
            
            # 检测爆炸
            frame = self.detect_explosion(frame)
            # 检测爆炸结束
            frame = self.detect_explosion_end(frame)
            
            # 创建显示帧
            display_frame = self.create_display_frame(frame)
            
            # 显示结果
            cv2.imshow("Missile Strike Analysis", display_frame)
            # 写入视频
            if output_path:
                out.write(display_frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 按'q'退出
                break
            elif key == ord('s') and output_path:  # 按's'保存当前帧为JPEG图片
                frame_path = f"frame_{self.frame_counter}.jpg"
                cv2.imwrite(frame_path, display_frame)
                print(f"已保存当前帧到 {frame_path}")

        # 释放资源
        self.cap.release()
        # 关闭视频写入器，确保输出文件完整保存
        if output_path:
            out.release()
        # 销毁所有由 cv2.imshow() 创建的窗口
        cv2.destroyAllWindows()
        # 返回检测到的关键帧及其对应时间戳
        return self.key_frames, self.key_frames_time

class MissileStrikeAnalyzerCamera(MissileStrikeAnalyzer):

    # 继承自 MissileStrikeAnalyzer，专用于摄像头实时视频分析。尝试打开指定索引的摄像头，失败则抛出异常
    def __init__(self, camera_index=0):
        # 使用摄像头捕获视频
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise ValueError("无法打开摄像头")

        self.explosion_change_ratio_threshold = 0.2  # 爆炸变化比例阈值
        # 获取摄像头的帧率和分辨率
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 由于摄像头是实时流，总帧数不适用
        self.total_frames = float('inf') # 实时流无固定帧数

        # 初始化其他参数
        self.display_width = self.frame_width * 2
        self.display_height = self.frame_height
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

        self.feature_params = dict(  # Shi-Tomasi角点检测参数
            maxCorners=100,          # 最大角点数量
            qualityLevel=0.3,        # 角点质量阈值（0-1）
            minDistance=7,          # 角点间最小像素距离
            blockSize=7              # 计算区域大小
        )
        self.lk_params = dict(       # Lucas-Kanade光流参数
            winSize=(15,15),         # 搜索窗口大小
            maxLevel=2,              # 金字塔层数
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.prev_gray = None        # 前一帧灰度图
        self.prev_points = None     # 前一帧跟踪点
        self.missile_trajectory = [] # 导弹轨迹记录

        self.missile_bbox = None    # 导弹边界框
        self.explosion_threshold = 200 # 爆炸亮度阈值
        self.explosion_min_area = self.frame_width * self.frame_height * 0.01  # 爆炸最小区域
        # 状态标志
        self.missile_detected = False
        self.impact_detected = False
        self.explosion_detected = False
        self.explosion_end_detected = False

        self.frame_counter = 0            # 当前帧计数器
        self.impact_frame_count = 0       # 碰撞后冷却帧计数
        self.cooldown_frames = int(self.fps * 1.0)  # 碰撞后1秒内不重复检测
        self.previous_frames = []         # 缓存前N帧用于差分分析
        self.frames_to_keep = 5           # 缓存帧数
        self.brightness_threshold = 30
        self.color_change_threshold = 40
        self.stability_threshold = 15
        self.stability_counter = 0
        self.pixel_stability_threshold = 5
        self.previous_explosion_frames = []  # 爆炸帧缓存
        self.explosion_frames_to_keep = 10   # 爆炸分析缓存大小
        self.min_contour_area_ratio = 0.002  # 最小轮廓面积比

        # === 在这里添加这一行 ===
        self.min_roi_pixels = 1024 # 32*32
        # =====================

        # 中文字体路径探测
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

def main():
    # 创建命令行参数解析器，设置系统描述
    parser = argparse.ArgumentParser(description='导弹打击目标视觉识别系统')
    # 添加命令行参数
    parser.add_argument('--video_path', type=str, default="Mk82.mp4",required=False, help='输入视频的路径')
    parser.add_argument('--camera', action='store_true', help='使用摄像头模式')
    parser.add_argument('--output', type=str, default=None, help='输出视频的路径')
    # 解析命令行参数
    args = parser.parse_args()

    # 参数校验：必须提供视频路径或启用摄像头模式
    if not args.camera and not args.video_path:
        parser.error("必须提供 --video_path 或使用 --camera")
    if args.camera and args.video_path and args.video_path != "Mk82.mp4": # 允许默认值和camera同时存在，但用户指定了video_path则冲突
         parser.error("--camera 和 --video_path 不能同时使用")


    # 根据参数选择初始化分析器
    if args.camera:
        print("使用摄像头模式...")
        try:
            analyzer = MissileStrikeAnalyzerCamera()  # 摄像头模式：实例化摄像头分析器类
        except ValueError as e:
            print(f"错误: {e}")
            return
    else:
        print(f"分析视频文件: {args.video_path}")
        if not os.path.exists(args.video_path):
             print(f"错误: 视频文件 '{args.video_path}' 不存在!")
             return
        try:
            analyzer = MissileStrikeAnalyzer(args.video_path)  # 视频文件模式：传入视频路径实例化分析器
        except ValueError as e:
            print(f"错误: {e}")
            return


    # 处理视频/摄像头流，返回关键帧和时间数据
    key_frames, key_frames_time = analyzer.process_video(args.output)

    # 打印检测结果（保持不变）
    print("\n检测到的关键事件:")
    for event_name, time in key_frames_time.items():
        # 事件名称中英映射
        event_display = {
            "missile_appearance": "导弹出现",
            "missile_impact": "导弹接触目标",
            "explosion": "爆炸发生",
            "explosion_end": "爆炸结束"
        }
        # 格式化输出时间或未检测提示
        if time is not None:
            print(f"{event_display[event_name]}: {analyzer.format_time(time)}")
        else:
            print(f"{event_display[event_name]}: 未检测到")

if __name__ == "__main__":
    main()


# ========== Added: Headless, fast wrapper API ==========
import os
import cv2

def ML_based_Approach(video_path, target_roi_xywh):
    """
    无UI、全速、导出四张“干净”关键帧的封装入口。
    返回固定顺序：DD出现/接触/点火/结束。
    """
    # --- 校验 ROI ---
    if (not isinstance(target_roi_xywh, (tuple, list))) or len(target_roi_xywh) != 4:
        raise ValueError("target_roi_xywh 必须是长度为4的 (x, y, w, h) 元组/列表")
    x, y, w, h = [int(v) for v in target_roi_xywh]
    if w <= 0 or h <= 0:
        raise ValueError("ROI 宽高必须 > 0")

    # --- 初始化分析器 ---
    analyzer = MissileStrikeAnalyzer(video_path)
    analyzer.target_roi = (x, y, w, h)
    if hasattr(analyzer, "select_target_roi"):
        analyzer.select_target_roi = lambda: None  # 跳过交互式ROI

    # --- 无界面、非原速 ---
    _orig_imshow, _orig_waitKey, _orig_destroyAll = cv2.imshow, cv2.waitKey, cv2.destroyAllWindows
    try:
        cv2.imshow = lambda *args, **kwargs: None
        cv2.waitKey = lambda *args, **kwargs: -1
        cv2.destroyAllWindows = lambda *args, **kwargs: None
        key_frames, key_times = analyzer.process_video(output_path=None)
    finally:
        cv2.imshow, cv2.waitKey, cv2.destroyAllWindows = _orig_imshow, _orig_waitKey, _orig_destroyAll

    # --- 重新抓“干净帧”并保存（修正命名：去掉扩展名） ---
    abs_video_path = os.path.abspath(video_path)               # e.g. D:/tmp/1-1.mp4
    abs_video_base, _ = os.path.splitext(abs_video_path)       # e.g. D:/tmp/1-1   <- 用这个来命名

    def _fmt_hhmmss(seconds):
        if seconds is None: return "未检测到"
        if seconds < 0: seconds = 0
        total = int(round(seconds))
        hh = total // 3600
        mm = (total % 3600) // 60
        ss = total % 60
        return f"{hh:02d}.{mm:02d}.{ss:02d}"

    def _grab_clean_frame(vpath, t_seconds):
        if t_seconds is None: return None
        cap2 = cv2.VideoCapture(vpath)
        if not cap2.isOpened(): return None
        cap2.set(cv2.CAP_PROP_POS_MSEC, max(t_seconds * 1000.0 - 5.0, 0.0))
        ok, frm = cap2.read()
        if not ok or frm is None:
            fps = cap2.get(cv2.CAP_PROP_FPS) or 0
            if fps > 0:
                frame_id = int(round(max(t_seconds * fps - 1, 0)))
                cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, frm = cap2.read()
        cap2.release()
        return frm if ok and frm is not None else None

    event_map = [
        ("missile_appearance", "导弹出现", 1),
        ("missile_impact",     "导弹接触目标",   2),
        ("explosion",          "导弹爆炸",   3),
        ("explosion_end",      "爆炸结束",   4),
    ]

    res = []
    for key, cn_name, idx in event_map:
        t = (key_times or {}).get(key, None)
        clean = _grab_clean_frame(abs_video_path, t)
        if clean is not None:
            # 关键：用 abs_video_base（去除扩展名）
            capture_path = f"{abs_video_base}.{idx}.png"  # e.g. D:/tmp/1-1.1.png
            try:
                cv2.imwrite(capture_path, clean)
            except Exception:
                # 回退：保存到当前工作目录，仍然使用去扩展名后的文件名
                capture_path = os.path.abspath(f"./{os.path.basename(abs_video_base)}.{idx}.png")
                cv2.imwrite(capture_path, clean)
            time_str = _fmt_hhmmss(t)
        else:
            capture_path = ""
            time_str = "未检测到"

        res.append({
            "capture": capture_path,
            "time": time_str,
            "eventName": cn_name,
        })

    return res

    def _fmt_hhmmss(seconds):
        if seconds is None:
            return "未检测到"
        if seconds < 0:
            seconds = 0
        total = int(round(seconds))
        hh = total // 3600
        mm = (total % 3600) // 60
        ss = total % 60
        return f"{hh:02d}.{mm:02d}.{ss:02d}"

    def _grab_clean_frame(vpath, t_seconds):
        """
        从源视频按时间点抓取一帧（不走任何绘制/显示管线，保证无叠加）。
        先用 POS_MSEC 定位；失败则用 FPS 估算帧号定位。
        """
        if t_seconds is None:
            return None
        cap2 = cv2.VideoCapture(vpath)
        if not cap2.isOpened():
            return None

        ok = False
        frame = None
        # 优先用毫秒定位
        cap2.set(cv2.CAP_PROP_POS_MSEC, max(t_seconds * 1000.0 - 5.0, 0.0))
        ok, frm = cap2.read()
        if ok and frm is not None:
            frame = frm
        else:
            # 回退：用 FPS 计算帧号定位
            fps = cap2.get(cv2.CAP_PROP_FPS) or 0
            if fps > 0:
                frame_id = int(round(max(t_seconds * fps - 1, 0)))
                cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, frm = cap2.read()
                if ok and frm is not None:
                    frame = frm
        cap2.release()
        return frame

    # 事件顺序与命名
    event_map = [
        ("missile_appearance", "导弹出现", 1),
        ("missile_impact",     "导弹接触目标",   2),
        ("explosion",          "导弹爆炸",   3),
        ("explosion_end",      "爆炸结束",   4),
    ]

    res = []
    for key, cn_name, idx in event_map:
        t = (key_times or {}).get(key, None)
        clean = _grab_clean_frame(abs_video_path, t)
        if clean is not None:
            capture_path = f"{abs_video_path}.{idx}.png"
            try:
                cv2.imwrite(capture_path, clean)
            except Exception:
                # 回退：保存到当前工作目录
                capture_path = os.path.abspath(f"./{os.path.basename(abs_video_path)}\b\b\b\b.{idx}.png")
                cv2.imwrite(capture_path, clean)
            time_str = _fmt_hhmmss(t)
        else:
            capture_path = ""
            time_str = "未检测到"

        res.append({
            "capture": capture_path,
            "time": time_str,
            "eventName": cn_name,
        })

    return res
# ========== End of added wrapper API ==========
