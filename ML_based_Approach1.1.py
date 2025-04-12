#准备在1上直接进行改进

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
        self.cooldown_frames = int(self.fps * 1.0)  # 碰撞后 1 秒内不检测导弹
        # 以前的帧，用于爆炸检测
        self.previous_frames = []
        self.frames_to_keep = 5  # 保存多少帧用于爆炸检测
        # 爆炸亮度和颜色变化阈值
        self.brightness_threshold = 30
        self.color_change_threshold = 40

        # 爆炸结束检测参数
        self.stability_threshold = 30  # 连续稳定帧数阈值，增加到30帧以确保真正稳定
        self.stability_counter = 0  # 稳定帧计数器
        self.pixel_stability_threshold = 5  # 像素稳定性阈值，降低到5以捕获微小变化
        self.previous_explosion_frames = []  # 存储爆炸后的几帧用于分析稳定性
        self.explosion_frames_to_keep = 10  # 增加到10帧以进行更可靠的稳定性分析
        self.min_contour_area_ratio = 0.002  # 显著变化区域的最小面积比例，降低到目标区域的0.2%

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

    def select_target_roi(self):    #选择目标区域（重要参数：showCrosshair=True）
        # 读取第一帧
        ret, first_frame = self.cap.read()
        if not ret:
            raise ValueError("无法读取视频的第一帧")
            # 在第一帧上添加中文提示
        frame_with_text = self.put_chinese_text(first_frame, "请选择目标区域，按回车确认", (20, 40), 36, (0, 255, 255))
        # 显示第一帧并让用户选择目标区域
        cv2.namedWindow("Select Target Roi")
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

    def detect_missile(self, frame, fg_mask):   #检测并跟踪导弹（完成的核心步骤：碰撞冷却检测：跳过爆炸后的无效检测。​前景区域提取：通过二值化和形态学操作去噪。​轮廓定位：为后续导弹筛选（面积、长宽比、轨迹连续性检查）提供候选区域）
        """使用背景减除和光流检测导弹"""
        # 如果已经检测到碰撞，并且在冷却期间，不检测导弹(避免在爆炸后误检残骸为导弹)
        if self.impact_detected and self.impact_frame_count < self.cooldown_frames:
            self.impact_frame_count += 1    #递增冷却计数器
            self.missile_bbox = None  # 清除导弹边界框
            return frame    #直接返回当前帧，跳过后续检测逻辑
        # 将当前帧转为灰度图供光流法使用(光流算法需要单通道灰度图像进行像素运动分析)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        '''应用背景减除(对背景减除后的前景掩膜 fg_mask 进行二值化;）
        127是阈值，大于此值的像素设为 255（白色），否则为 0（黑色）；cv2.THRESH_BINARY：二值化模式。（固定阈值可能不适应光照变化场景，可以改用自适应阈值（如cv2.THRESH_OTSU）
        ​作用：将前景区域（移动物体）标记为白色，背景为黑色)'''
        moving_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)[1]
        '''功能：通过形态学操作去除噪声。
        ​开运算​（MORPH_OPEN）：先腐蚀后膨胀，消除小噪声点。闭运算​（MORPH_CLOSE）：先膨胀后腐蚀，填充前景区域的小孔洞。
        ​参数：kernel：5x5 的矩形结构元素，控制形态学操作的强度（较大核可能过度平滑小目标，可以根据视频分辨率动态调整核大小）
        效果：得到更干净的前景区域（导弹候选区域）'''
        kernel = np.ones((5, 5), np.uint8)
        moving_mask = cv2.morphologyEx(moving_mask, cv2.MORPH_OPEN, kernel)
        moving_mask = cv2.morphologyEx(moving_mask, cv2.MORPH_CLOSE, kernel)
        '''功能：在形态学处理后的掩膜上检测轮廓。
        ​参数：cv2.RETR_EXTERNAL：仅检测最外层轮廓（忽略内部孔洞）。cv2.CHAIN_APPROX_SIMPLE：压缩冗余轮廓点（例如直线只保留端点）。
        ​输出：contours：轮廓列表，每个轮廓由一系列点组成。
        ​目的：定位前景区域的位置和形状，供后续导弹筛选使用。'''
        contours, _ = cv2.findContours(moving_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 检测可能的导弹（初始化导弹检测标志、边界框和中心点）
        missile_detected = False
        self.missile_bbox = None
        missile_center = None
        # 如果已经检测到碰撞和爆炸，不再检测新的导弹
        if self.impact_detected and self.explosion_detected:
            return frame
        for contour in contours:
            # 计算轮廓的面积和边界框(功能：遍历所有前景轮廓，通过面积过滤候选目标)
            area = cv2.contourArea(contour)
            # 过滤小的噪点和大的区域
            if 50 < area < 2000:  # 调整这些阈值以适应导弹大小（50和2000是固定值，可能不适用于不同分辨率或场景的视频）
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                # 导弹通常是细长的（功能：通过长宽比筛选细长目标；）
                '''若导弹外形非细长（如球形弹头），会导致漏检
                优化建议：
                # 结合其他形状特征（如轮廓最小外接矩形方向）
                rect = cv2.minAreaRect(contour)
                (_, _), (width, height), angle = rect
                aspect_ratio = max(width, height) / min(width, height)
                if aspect_ratio > 3:  # 更宽松的长宽比阈值'''
                if 0.1 < aspect_ratio < 0.5:
                    # 如果已经检测到碰撞，提高标准
                    if self.impact_detected:
                        continue
                        # 碰撞后不再检测导弹
                        '''若视频中存在多枚导弹，爆炸后无法检测后续目标
                        优化建议：改为基于时间的冷却机制（例如爆炸后N帧内不检测）
                    if self.impact_detected and (self.frame_counter - self.last_impact_frame) < self.cooldown_frames:
                        continue'''
                    missile_center = (int(x + w / 2), int(y + h / 2))
                    # 轨迹验证: 检查运动是否一致（通过距离阈值验证运动连续性）
                    if len(self.missile_trajectory) > 0:
                        last_center = self.missile_trajectory[-1]
                        dx = missile_center[0] - last_center[0]
                        dy = missile_center[1] - last_center[1]
                        distance = np.sqrt(dx * dx + dy * dy)
                        # 如果移动距离太远，可能不是同一个导弹
                        ''' 50像素的阈值无法适应不同速度或帧率
                            # 动态计算阈值（例如根据帧率和预估最大速度）
                            fps = self.cap.get(cv2.CAP_PROP_FPS)
                            max_speed_pixels_per_second = 1000  # 假设导弹最大速度
                            max_distance_per_frame = max_speed_pixels_per_second / fps
                            if distance > max_distance_per_frame * 1.5:  # 1.5倍容差
                                continue'''
                        if distance > 50:  # 距离阈值50可能需根据帧率和导弹速度调整。
                            continue
                    self.missile_trajectory.append(missile_center)
                    self.missile_bbox = (x, y, w, h)
                    # 如果我们检测到连续几帧中有导弹，则确认导弹出现
                    '''3帧的确认阈值可能导致低速导弹检测延迟
                    优化建议：# 使用加权评分系统（例如面积+长宽比+轨迹一致性）'''
                    # confidence_score = 0
                    # confidence_score += area / max_area  # 面积得分
                    # confidence_score += aspect_ratio / 3  # 长宽比得分（假设最大为3:1）
                    # if len(self.missile_trajectory) > 0:
                    #    # 轨迹平滑度得分（距离越小得分越高）
                    #    confidence_score += 1 - min(distance / max_distance_per_frame, 1)
                    # if confidence_score > 2.0:  # 自定义阈值
                    #     missile_detected = True
                    if len(self.missile_trajectory) > 3:
                        missile_detected = True
                        # 在帧上标记导弹
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # 一旦找到导弹，就中断循环，避免多重检测
                    break
                    # 如果检测到导弹，保存关键帧
                    '''break会导致只检测第一个符合条件的轮廓，漏检其他目标
                    优化建议：# 收集所有候选目标，最后选择最优
                    candidates = []
                    for contour in contours:
                        # ...（筛选逻辑）
                        if valid:
                            candidates.append( (contour, confidence_score) )
                    # 选择置信度最高的候选
                    if candidates:
                        best_contour = max(candidates, key=lambda x: x[1])[0]
                        x, y, w, h = cv2.boundingRect(best_contour)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                    '''
        '''功能：首次检测触发保存：当当前帧检测到导弹（missile_detected=True）且之前未检测到（self.missile_detected=False）时：
        标记导弹已检测到（self.missile_detected=True）。
        保存当前帧的副本到 key_frames 字典。
        记录时间戳（基于帧计数器 frame_counter 和视频帧率 fps）。
        ​潜在问题：
        ​单次保存：仅在首次检测时保存关键帧，后续导弹出现不会触发。
        ​内存消耗：frame.copy() 可能占用较大内存（尤其是高分辨率视频）
        # 添加间隔机制（例如每30帧保存一次）
        if missile_detected and (self.frame_counter - self.last_missile_frame) > 30:
            self.key_frames[f"missile_{self.frame_counter}"] = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]'''
        if missile_detected and not self.missile_detected:
            self.missile_detected = True
            self.key_frames["missile_appearance"] = frame.copy()
            self.key_frames_time["missile_appearance"] = self.frame_counter / self.fps

            # 更新光流跟踪
        '''功能：初始化光流跟踪所需的前一帧灰度图（仅在第一帧执行）。
        ​潜在问题：未处理视频流重新开始的情况（如循环播放视频）。
        ​改进建议：
        # 在视频重置时清空 prev_gray
        if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == 0:
            self.prev_gray = None'''
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            # 如果我们已经有了跟踪点，更新它们
            '''功能：
        使用 Lucas-Kanade 光流法计算当前帧中跟踪点的位置。
        ​参数说明：
        self.lk_params：光流法参数（如 winSize=(15,15) 定义搜索窗口大小）。
        status：标记每个点是否成功跟踪（1成功，0失败）。
        ​潜在问题：
        ​累积误差：长期跟踪可能导致特征点漂移
        改进建议：
        # 定期重新检测特征点（例如每10帧）
        if self.frame_counter % 10 == 0:
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=moving_mask, ​**self.feature_params)'''
        if self.prev_points is not None and len(self.prev_points) > 0:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_points, None, **self.lk_params)
            # 选择好的点
            '''功能：
            保留成功跟踪的点（status=1），过滤丢失的点。
            ​潜在问题：
            若所有点丢失（good_next 为空），光流跟踪会中断。
            ​改进建议：
            if next_points is not None and len(good_next) > 0:
                # 保留至少5个点以确保跟踪稳定性
                if len(good_next) < 5:
                    self.prev_points = cv2.goodFeaturesToTrack(gray, mask=moving_mask, ​**self.feature_params)'''
            if next_points is not None:
                good_next = next_points[status == 1]
                good_prev = self.prev_points[status == 1]
                # 绘制跟踪线
                '''功能：
                在非碰撞状态下，绘制绿色线段表示光流轨迹。
                ​潜在问题：
                ​绘制性能：大量线段绘制可能影响实时性。
                ​改进建议：
                # 限制绘制的轨迹数量（例如最多20条）
                for i, (new, old) in enumerate(zip(good_next[:20], good_prev[:20])):'''
                for i, (new, old) in enumerate(zip(good_next, good_prev)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    # 只绘制非碰撞区域的光流线
                    if not self.impact_detected:
                        frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                '''功能：
                如果有有效跟踪点，更新为当前帧的点；否则重新检测新点。
                ​参数说明：
                cv2.goodFeaturesToTrack：使用 Shi-Tomasi 角点检测算法，参数包括：
                maxCorners=100：最大特征点数量。
                qualityLevel=0.3：最低特征值阈值（相对于最大特征值的比例）。
                minDistance=7：特征点之间的最小像素距离。
                ​潜在问题：
                ​特征点质量：qualityLevel=0.3 可能检测到噪声点。
                ​改进建议：
                python
                # 提高质量阈值并限制检测区域
                self.feature_params = dict(
                    maxCorners=50,
                    qualityLevel=0.5,
                    minDistance=10,
                    blockSize=7,
                    mask=moving_mask  # 仅在运动区域内检测
                )'''
                self.prev_points = good_next.reshape(-1, 1, 2)
        else:
            # 如果没有跟踪点，尝试检测新的点
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=moving_mask, **self.feature_params)
        #保存当前帧灰度图供下一帧光流计算使用，并返回标记后的帧
        self.prev_gray = gray.copy()
        return frame

    def detect_impact(self, frame): #检测导弹与目标的接触（关键技术：区域重叠检测；使用check_rectangles_overlap方法）
        
        """检测导弹与目标的接触"""
        '''功能：
        仅在以下条件满足时触发碰撞检测：
        导弹已被检测到（self.missile_detected=True）。
        尚未检测到碰撞（self.impact_detected=False）。
        导弹的边界框存在（self.missile_bbox is not None）。
        ​潜在问题：
        未处理导弹消失后边界框为 None 的情况（可能导致后续逻辑错误）。
        ​优化建议：
        # 添加防御性检查
        if self.missile_detected and not self.impact_detected:
            if self.missile_bbox is None:
                return frame'''
        if self.missile_detected and not self.impact_detected and self.missile_bbox is not None:
            
            # 检查导弹边界框是否与目标区域重叠
            '''功能：
            调用 check_rectangles_overlap 方法，判断导弹边界框（missile_bbox）与目标区域（target_roi）是否重叠。
            ​依赖方法：
            check_rectangles_overlap 需正确处理矩形坐标比较（如边界接触是否算重叠）。
            ​潜在问题：
            如果矩形坐标为负或超出画面范围，可能导致逻辑错误。
            ​优化建议：
            # 添加矩形坐标合法性检查
            def is_valid_rect(rect):
                x, y, w, h = rect
                return w > 0 and h > 0 and x >= 0 and y >= 0
            if is_valid_rect(self.missile_bbox) and is_valid_rect(self.target_roi):
                # 执行重叠检测'''
            if self.check_rectangles_overlap(self.missile_bbox, self.target_roi):
                
                '''功能：
                标记碰撞已发生（impact_detected=True）。
                重置碰撞冷却计数器（impact_frame_count=0）。
                保存当前帧和时间戳到关键帧字典。
                ​潜在问题：
                frame.copy() 可能占用大量内存（高分辨率视频）。
                ​优化建议：
                # 使用图像压缩保存关键帧
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                self.key_frames["missile_impact"] = buffer.tobytes()'''
                self.impact_detected = True
                self.impact_frame_count = 0  # 重置碰撞后计数器
                self.key_frames["missile_impact"] = frame.copy()
                self.key_frames_time["missile_impact"] = self.frame_counter / self.fps
                
                # 在碰撞位置绘制黄色碰撞标记
                '''功能：
                计算导弹与目标区域的重叠区域坐标和尺寸。
                ​数学逻辑：
                overlap_x 和 overlap_y：重叠区域的左上角坐标（取两个矩形左/上边界的较大值）。
                overlap_w 和 overlap_h：重叠区域的宽高（取右/下边界的较小值减去左上角坐标）。
                ​潜在问题：
                若两个矩形无重叠，计算结果可能为负值（但已通过 check_rectangles_overlap 检查，此处可忽略）。
                ​优化建议：
                # 添加非负检查（防御性编程）
                overlap_w = max(0, overlap_w)
                overlap_h = max(0, overlap_h)'''
                missile_x, missile_y, missile_w, missile_h = self.missile_bbox
                target_x, target_y, target_w, target_h = self.target_roi
                # 计算重叠区域
                overlap_x = max(missile_x, target_x)
                overlap_y = max(missile_y, target_y)
                overlap_w = min(missile_x + missile_w, target_x + target_w) - overlap_x
                overlap_h = min(missile_y + missile_h, target_y + target_h) - overlap_y
                
                # 在重叠区域绘制黄色标记
                '''功能：
                在重叠区域填充黄色矩形（(0, 255, 255) 为 BGR 黄色，-1 表示填充）。
                ​潜在问题：
                若重叠区域尺寸为0（如边界接触），绘制无意义。
                ​优化建议：
                if overlap_w > 0 and overlap_h > 0:
                    cv2.rectangle(frame, (overlap_x, overlap_y), 
                                (overlap_x + overlap_w, overlap_y + overlap_h), 
                                (0, 255, 255), -1)'''
                cv2.rectangle(frame, (overlap_x, overlap_y),
                              (overlap_x + overlap_w, overlap_y + overlap_h),
                              (0, 255, 255), -1)  # 填充重叠区域
                # 在原始边界框上方添加碰撞文字
                text_pos = (target_x, target_y - 10)
                frame = self.put_chinese_text(frame, "碰撞!", text_pos, 30, (0, 0, 255))

        # 绘制目标区域
        '''功能：
        ​条件检查：如果目标区域 self.target_roi 存在（非空或非None），则绘制目标区域。
        ​坐标解包：提取目标区域的左上角坐标 (x, y) 和宽高 (w, h)。
        ​绘制蓝色矩形：使用 OpenCV 的 cv2.rectangle 在图像上绘制蓝色边框（BGR 颜色值 (255, 0, 0) 表示蓝色），线宽为 2 像素。
        ​潜在问题：
        ​无效矩形参数：如果 w 或 h 为负数或零，会导致绘制错误。
        ​坐标越界：如果 x + w 或 y + h 超出图像尺寸，可能导致绘制异常。
        ​优化建议：
        if self.target_roi:
            x, y, w, h = self.target_roi
            # 检查矩形参数有效性
            if w > 0 and h > 0:
                # 确保坐标在图像范围内
                img_h, img_w = frame.shape[:2]
                x2 = max(0, min(x + w, img_w))
                y2 = max(0, min(y + h, img_h))
                x1 = max(0, min(x, img_w))
                y1 = max(0, min(y, img_h))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)'''
        if self.target_roi:
            x, y, w, h = self.target_roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # 绘制导弹区域 - 只在未检测到碰撞时显示
        '''功能：
        ​条件检查：仅在导弹边界框 self.missile_bbox 存在且未检测到碰撞时绘制。
        ​坐标解包：提取导弹区域的左上角坐标 (x, y) 和宽高 (w, h)。
        ​绘制绿色矩形：使用绿色边框（BGR 颜色值 (0, 255, 0)），线宽为 2 像素。
        ​潜在问题：
        ​碰撞后导弹消失：碰撞后导弹框隐藏，可能让用户无法追踪导弹的最终位置。
        ​缺乏状态反馈：仅隐藏框体，缺乏碰撞后的视觉反馈（如颜色变化）。
        ​优化建议：
        if self.missile_bbox:
            x, y, w, h = self.missile_bbox
            if w > 0 and h > 0:
                # 根据碰撞状态选择颜色
                color = (0, 255, 0) if not self.impact_detected else (0, 0, 255)  # 碰撞后变红色
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                # 碰撞后添加额外标记（如文字或符号）
                if self.impact_detected:
                    cv2.putText(frame, "HIT", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)'''
        if self.missile_bbox and not self.impact_detected:
            x, y, w, h = self.missile_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame

    def detect_explosion(self, frame):  #检测爆炸和烟尘发生（关键技术：HSV颜色空间分析；重要参数：brightness_threshold=30）
        """检测爆炸和烟尘"""
        # 如果已经检测到爆炸，直接返回当前帧，避免重复计算
        '''潜在问题：
        ​缺少爆炸标记：若需在帧上绘制爆炸效果（如红框或文字），此处直接返回原始帧会导致视觉反馈缺失。
        ​优化建议：
        if self.explosion_detected:
            # 在帧上绘制爆炸标记（例如红色文字）
            cv2.putText(frame, "EXPLOSION", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame'''
        if self.explosion_detected:
            return frame
        # 如果未检测到碰撞，不检测爆炸
        '''功能：
        缓存最近 self.frames_to_keep 帧图像，用于后续爆炸检测（如对比碰撞前后的亮度变化）。
        若未发生碰撞，直接返回当前帧。
        ​潜在问题：
        ​低效的列表操作：pop(0) 的时间复杂度为 O(N)，当 frames_to_keep 较大时性能差。
        ​未处理初始空缓存：若首次调用时 previous_frames 为空，可能导致后续逻辑错误。
        ​优化建议：
        from collections import deque
        # 在类初始化中定义双端队列
        def __init__(self):
            self.previous_frames = deque(maxlen=self.frames_to_keep)  # 自动维护长度
        # 方法中直接追加，无需手动检查长度
        if not self.impact_detected:
            self.previous_frames.append(frame.copy())
            return frame'''
        if not self.impact_detected:
            # 保存当前帧用于后续爆炸检测
            if len(self.previous_frames) >= self.frames_to_keep:
                self.previous_frames.pop(0)
            self.previous_frames.append(frame.copy())
            return frame
        # 提取目标区域进行分析
        '''功能：从当前帧中提取目标区域的图像块。
        ​关键问题：
        ​未验证目标区域有效性：若 self.target_roi 未定义或坐标越界，会引发 ValueError。
        ​越界切片风险：如 target_y + target_h 超出图像高度，导致 target_region 为空数组。
        ​优化建议：
        python
        # 检查目标区域是否存在
        if not hasattr(self, 'target_roi') or not self.target_roi:
            return frame
        # 提取并验证坐标
        target_x, target_y, target_w, target_h = self.target_roi
        img_h, img_w = frame.shape[:2]
        # 调整坐标和尺寸至图像范围内
        x1 = max(0, min(target_x, img_w - 1))
        y1 = max(0, min(target_y, img_h - 1))
        x2 = max(0, min(target_x + target_w, img_w))
        y2 = max(0, min(target_y + target_h, img_h))
        target_region = frame[y1:y2, x1:x2]
        # 检查区域是否有效
        if target_region.size == 0:
            return frame'''
        target_x, target_y, target_w, target_h = self.target_roi
        target_region = frame[target_y:target_y + target_h, target_x:target_x + target_w]

        # 如果没有足够的历史帧，返回
        '''功能：确保至少有2帧历史数据用于对比，避免单帧无法计算差异。
        ​潜在问题：
        ​缓存策略低效：若使用列表的 pop(0) 维护历史帧，性能较差（时间复杂度 O(N)）。
        ​未处理空帧：若 self.previous_frames 中存在空帧，可能导致后续操作崩溃。
        ​优化建议：
        # 使用 deque 优化缓存性能（在类初始化中定义）
        from collections import deque
        self.previous_frames = deque(maxlen=5)  # 根据需求调整 maxlen
        # 检查有效帧数量
        if len(self.previous_frames) < 2 or frame is None:
            return frame'''
        if len(self.previous_frames) < 2:
            return frame
        
        # 获取碰撞前的帧
        '''功能：从历史帧中获取碰撞前的目标区域图像块。
        ​关键问题：
        ​坐标越界风险：未校验目标区域是否在 pre_impact_frame 的合法范围内。
        ​硬编码索引依赖：假设 self.previous_frames[0] 是碰撞前的关键帧，需确保缓存逻辑正确。
        ​优化建议：
        # 使用动态坐标修正（参考前文代码）
        x1, y1, x2, y2 = self._get_safe_roi_coordinates(target_x, target_y, target_w, target_h, frame.shape)
        # 提取区域前检查历史帧是否有效
        pre_impact_frame = self.previous_frames[-1]  # 更合理的索引（假设最新帧是碰撞前最后一帧）
        if pre_impact_frame is None:
            return frame
        pre_impact_region = pre_impact_frame[y1:y2, x1:x2]'''
        pre_impact_frame = self.previous_frames[0]
        pre_impact_region = pre_impact_frame[target_y:target_y + target_h, target_x:target_x + target_w]
        
        # 转换为 HSV 色彩空间以分析颜色变化
        '''功能：将当前帧和碰撞前帧的RGB图像转为HSV空间，分别计算H（色调）、S（饱和度）、V（亮度）通道的绝对差异。
        ​潜在问题：
        ​通道权重相同：H/S/V三个通道的差异被等权重处理，但实际爆炸中亮度（V）变化可能更重要。
        ​未归一化差异：H通道范围为0-180（OpenCV约定），S/V为0-255，直接比较可能导致H通道灵敏度不足。
        ​优化建议：
        # 归一化H通道差异（H范围是0-180，需缩放到0-255）
        h_diff = cv2.absdiff(hsv_current[:, :, 0], hsv_pre[:, :, 0]).astype(np.float32)
        h_diff = (h_diff / 180 * 255).astype(np.uint8)  # 归一化到0-255范围
        # 加权差异（例如亮度权重更高）
        combined_diff = 0.2 * h_diff + 0.3 * s_diff + 0.5 * v_diff
        combined_diff = combined_diff.astype(np.uint8)'''
        hsv_current = cv2.cvtColor(target_region, cv2.COLOR_BGR2HSV)
        hsv_pre = cv2.cvtColor(pre_impact_region, cv2.COLOR_BGR2HSV)
        # 计算亮度和颜色差异
        h_diff = cv2.absdiff(hsv_current[:, :, 0], hsv_pre[:, :, 0])
        s_diff = cv2.absdiff(hsv_current[:, :, 1], hsv_pre[:, :, 1])
        v_diff = cv2.absdiff(hsv_current[:, :, 2], hsv_pre[:, :, 2])
        
        '''功能：
        对每个通道的差异进行二值化，合并成综合变化区域。
        通过开运算（去噪）和闭运算（填充空洞）优化二值化结果。
        ​潜在问题：
        ​固定阈值和内核大小：brightness_threshold 和 color_change_threshold 可能需动态调整；5x5内核可能不适用于小目标。
        ​通道合并策略简单：直接按位或操作可能引入噪声（例如H通道的微小变化被误判为爆炸）。
        ​优化建议：
        # 动态内核大小（基于目标区域尺寸）
        kernel_size = max(1, int(min(target_w, target_h) * 0.1))  # 内核大小为区域尺寸的10%
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # 改进通道合并策略（例如仅当V通道超过阈值且H/S中至少一个超过）
        v_valid = v_thresh > 0
        hs_valid = (h_thresh > 0) | (s_thresh > 0)
        combined_change = np.bitwise_and(v_valid, hs_valid).astype(np.uint8) * 255
        # 应用形态学操作
        combined_change = cv2.morphologyEx(combined_change, cv2.MORPH_OPEN, kernel)
        combined_change = cv2.morphologyEx(combined_change, cv2.MORPH_CLOSE, kernel)'''
        # 检测大幅度亮度变化（爆炸通常伴随亮度急剧增加）
        _, v_thresh = cv2.threshold(v_diff, self.brightness_threshold, 255, cv2.THRESH_BINARY)
        # 检测火焰和烟雾的颜色变化
        _, h_thresh = cv2.threshold(h_diff, self.color_change_threshold, 255, cv2.THRESH_BINARY)
        _, s_thresh = cv2.threshold(s_diff, self.color_change_threshold, 255, cv2.THRESH_BINARY)
        # 合并各种变化
        combined_change = cv2.bitwise_or(cv2.bitwise_or(h_thresh, s_thresh), v_thresh)
        # 应用形态学操作
        kernel = np.ones((5, 5), np.uint8)
        combined_change = cv2.morphologyEx(combined_change, cv2.MORPH_OPEN, kernel)
        combined_change = cv2.morphologyEx(combined_change, cv2.MORPH_CLOSE, kernel)
        
        '''功能：通过变化区域占比判断是否发生爆炸，若触发则在帧上绘制标记。
        ​关键问题：
        ​硬编码阈值（0.2）​：缺乏灵活性，应设为可配置参数。
        ​未处理目标区域尺寸变化：若目标区域非常小（如10x10），total_pixels=100，此时 white_pixels=20 即可触发爆炸，可能误检。
        ​中文显示依赖自定义方法：若 put_chinese_text 未正确处理中文，可能显示乱码。
        ​优化建议：
        # 将阈值设为类变量
        self.explosion_change_ratio_threshold = 0.2  # 在初始化中定义
        # 动态调整阈值（例如小目标区域需要更高比例）
        min_pixels = 1000  # 假设目标区域至少 32x32 (1024 pixels)
        adjusted_threshold = self.explosion_change_ratio_threshold
        if total_pixels < min_pixels:
            adjusted_threshold += (min_pixels - total_pixels) / min_pixels * 0.1  # 小目标阈值提高
        if change_ratio > adjusted_threshold:
            # 标记爆炸
            self.explosion_detected = True
            # 绘制更醒目标记（如闪烁效果）
            if (self.frame_counter // 5) % 2 == 0:  # 每5帧切换一次颜色
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            # 确保中文字体正确渲染（使用PIL库）
            frame = self._draw_chinese_text(frame, "爆炸", (x1, y1 - 40), (0, 0, 255), 30)'''
        # 计算变化区域
        white_pixels = cv2.countNonZero(combined_change)
        total_pixels = target_w * target_h
        change_ratio = white_pixels / total_pixels
        # 如果变化比例超过阈值，认为爆炸发生
        if change_ratio > 0.2:  # 调整这个阈值
            self.explosion_detected = True
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

    def detect_explosion_end(self, frame):  #检测爆炸结束（关键技术：多帧稳定性分析；重要参数：stability_threshold=30,需要连续30帧稳定才确认爆炸结束）
        """严格检测爆炸结束 - 目标区域完全恢复稳定"""
        
        '''功能：若已检测到爆炸结束，直接返回当前帧，避免重复计算。
        ​问题：缺乏视觉反馈（如绘制“爆炸结束”标记）。
        ​优化建议：
        if self.explosion_end_detected:
            # 添加爆炸结束标记（例如绿色边框）
            cv2.rectangle(frame, (target_x, target_y), (target_x+target_w, target_y+target_h), (0, 255, 0), 2)
            return frame'''
        # 如果已经检测到爆炸结束，直接返回
        if self.explosion_end_detected:
            return frame

        '''功能：仅在爆炸已触发时执行后续逻辑，避免无效计算。
        ​问题：未处理self.explosion_detected状态异常（如误触发后无法恢复）。
        ​优化建议：
        # 添加状态重置逻辑（例如超时自动重置）
        if not self.explosion_detected:
            self.explosion_end_detected = False  # 可选：同步重置结束标记
            return frame'''
        # 如果未检测到爆炸，不检测爆炸结束
        if not self.explosion_detected:
            return frame

        '''功能：从当前帧提取目标区域（ROI）。
        ​关键问题：
        ​坐标越界风险：未校验目标区域是否超出图像边界。
        ​空区域风险：若目标区域尺寸为0，后续操作会崩溃。
        ​优化建议：
        # 动态修正目标区域坐标
        img_h, img_w = frame.shape[:2]
        x1 = max(0, min(target_x, img_w - 1))
        y1 = max(0, min(target_y, img_h - 1))
        x2 = max(0, min(target_x + target_w, img_w))
        y2 = max(0, min(target_y + target_h, img_h))
        target_region = frame[y1:y2, x1:x2]
        # 检查空区域
        if target_region.size == 0:
            return frame'''
        # 提取目标区域进行分析
        target_x, target_y, target_w, target_h = self.target_roi
        target_region = frame[target_y:target_y + target_h, target_x:target_x + target_w]

        '''功能：维护一个固定长度的历史帧队列，用于多帧稳定性分析。
        ​关键问题：
        ​低效的列表操作：pop(0)时间复杂度为O(N)，队列较长时性能差。
        ​未使用深拷贝：copy()可能无法避免某些对象的引用问题。
        ​优化建议：
        from collections import deque
        # 在类初始化中定义双端队列
        def __init__(self):
            self.previous_explosion_frames = deque(maxlen=self.explosion_frames_to_keep)
        # 直接追加，无需手动管理长度
        self.previous_explosion_frames.append(target_region.copy())'''
        # 保存当前目标区域
        if len(self.previous_explosion_frames) >= self.explosion_frames_to_keep:
            self.previous_explosion_frames.pop(0)
        self.previous_explosion_frames.append(target_region.copy())

        '''功能：确保缓存的历史帧数量足够（如保存了 self.explosion_frames_to_keep 帧），否则直接返回当前帧。
        ​潜在问题：
        ​硬编码长度依赖：假设 self.explosion_frames_to_keep 必须等于 self.stability_threshold，否则逻辑不匹配。
        ​未使用高效数据结构：若用列表存储历史帧，pop(0) 性能较差。
        ​优化建议：
        # 使用 deque 自动维护队列长度（初始化时设置 maxlen=self.stability_threshold）
        from collections import deque
        self.previous_explosion_frames = deque(maxlen=self.stability_threshold)
        # 检查队列是否填满
        if len(self.previous_explosion_frames) < self.stability_threshold:
            return frame'''
        # 如果没有足够的历史帧，返回
        if len(self.previous_explosion_frames) < self.explosion_frames_to_keep:
            return frame

            # 获取当前帧和前一帧进行对比
        curr_frame = self.previous_explosion_frames[-1]
        prev_frame = self.previous_explosion_frames[-2]

        '''功能：计算两帧的绝对差异，并分离BGR三个通道的差异。
        ​潜在问题：
        ​未归一化差异：各通道差异未加权，可能忽略亮度（Y通道）的敏感性。
        ​优化建议：
        # 转为YUV空间，优先分析亮度通道（Y）
        yuv_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2YUV)
        yuv_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2YUV)
        y_diff, u_diff, v_diff = cv2.split(cv2.absdiff(yuv_curr, yuv_prev))
        # 加权亮度差异（Y通道权重更高）
        combined_diff = 0.6 * y_diff + 0.2 * u_diff + 0.2 * v_diff'''
        # 计算多通道帧间差异
        frame_diff = cv2.absdiff(curr_frame, prev_frame)
        # 分别检查每个颜色通道的变化
        b_diff, g_diff, r_diff = cv2.split(frame_diff)

        '''功能：对每个通道的差异二值化，合并所有通道的变化区域。
        ​关键问题：
        ​固定阈值适用性：所有通道共享同一阈值，可能无法适应不同场景。
        ​计算冗余：三次阈值操作和两次按位或操作可能影响性能。
        ​优化建议：
        # 使用单一通道（如亮度）或灰度差异
        gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray_diff = cv2.absdiff(gray_curr, gray_prev)
        _, combined_thresh = cv2.threshold(gray_diff, self.pixel_stability_threshold, 255, cv2.THRESH_BINARY)'''
        # 对每个通道应用阈值处理
        _, b_thresh = cv2.threshold(b_diff, self.pixel_stability_threshold, 255, cv2.THRESH_BINARY)
        _, g_thresh = cv2.threshold(g_diff, self.pixel_stability_threshold, 255, cv2.THRESH_BINARY)
        _, r_thresh = cv2.threshold(r_diff, self.pixel_stability_threshold, 255, cv2.THRESH_BINARY)
        # 合并所有通道的变化
        combined_thresh = cv2.bitwise_or(cv2.bitwise_or(b_thresh, g_thresh), r_thresh)

        '''功能：通过膨胀和闭运算连接相邻的变化区域，减少噪声干扰。
        ​潜在问题：
        ​固定核尺寸（7x7）​：可能对小目标区域过度平滑，或对大区域连接不足。
        ​优化建议：
        # 动态核尺寸（基于目标区域大小）
        kernel_size = max(3, int(min(target_w, target_h) * 0.1))  # 核大小为区域尺寸的10%
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel)'''
        # 应用形态学操作，连接相邻的变化区域
        kernel = np.ones((7, 7), np.uint8)  # 增大核尺寸以更好地连接相邻区域
        dilated = cv2.dilate(combined_thresh, kernel, iterations=1)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

        '''功能：检测面积超过阈值的连通区域，标记为显著变化。
        ​关键问题：
        ​面积计算误差：cv2.contourArea 对非闭合轮廓计算不准确。
        ​过早终止循环：break 在发现第一个大轮廓后即终止，可能漏检其他区域。
        ​优化建议：
        total_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            total_area += area
            if area > min_contour_area:
                cv2.drawContours(...)
        # 判断总变化面积是否超过阈值
        significant_changes = total_area > (target_w * target_h * 0.05)  # 总变化面积超过5%'''
        # 寻找连通区域
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 分析连通区域的大小
        significant_changes = False
        min_contour_area = target_w * target_h * self.min_contour_area_ratio  # 更小的阈值捕获更微小的变化
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_contour_area:
                significant_changes = True
                # 在原图上标记变化区域（可用于调试）
                cv2.drawContours(frame[target_y:target_y + target_h, target_x:target_x + target_w],
                                 [contour], -1, (0, 0, 255), 1)
                break

        '''功能：通过连续稳定帧数判断爆炸是否结束，并添加可视化反馈。
        ​潜在问题：
        ​计数器重置策略：任何一帧出现变化即重置计数器，可能导致检测延迟。
        ​中文显示兼容性：依赖 put_chinese_text 方法，若未正确处理字体可能显示乱码。
        ​优化建议：
        python
        # 使用抗锯齿字体（如PIL库）
        from PIL import ImageFont, ImageDraw, Image
        def put_chinese_text(self, frame, text, position, font_size, color):
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            font = ImageFont.truetype("simhei.ttf", font_size)
            draw.text(position, text, font=font, fill=color)
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)'''
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
        '''功能：将OpenCV的BGR图像转为RGB格式，并创建PIL的Image对象。
        关键点：OpenCV使用BGR通道顺序，而PIL使用RGB，需正确转换。
        ​潜在问题：输入图像可能非BGR格式（如灰度图），需确保img是3通道BGR格式。'''
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # 初始化PIL的绘图接口，用于在图像上绘制文本或图形
        draw = ImageDraw.Draw(img_pil)
        
        '''​功能：尝试加载用户指定的中文字体，若未指定则使用PIL默认字体。
        ​关键问题：
        ​默认字体不支持中文：load_default()加载的是PIL内置的位图字体，无法显示中文，会显示乱码或方框。
        ​字体路径有效性：若self.font_path无效（如文件不存在或格式错误），会抛出IOError。
        ​优化建议：
        设置备用字体路径列表，依次尝试加载。
        在类初始化时预加载字体，避免重复加载开销。
        def __init__(self):
            self.font = None
            self._load_font()
        def _load_font(self):
            font_paths = ["simhei.ttf", "arial.ttf", "/system/fonts/DroidSans.ttf"]  # 备用字体列表
            for path in font_paths:
                try:
                    self.font = ImageFont.truetype(path, default_size)
                    break
                except IOError:
                    continue
            if self.font is None:
                self.font = ImageFont.load_default()'''
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
        
        '''功能：若指定输出路径，创建视频写入对象，用于保存处理后的帧。
        ​关键参数：
        fourcc='XVID'：视频编码格式（兼容性较好）。
        self.fps：从输入视频中读取的帧率。
        (self.display_width, self.display_height)：输出视频的尺寸，需与 display_frame 尺寸一致。
        ​潜在问题：
        若 display_frame 尺寸与初始化参数不符，写入失败。
        ​优化建议：
        # 动态获取 display_frame 尺寸
        test_frame = self.create_display_frame(np.zeros_like(frame))
        display_height, display_width = test_frame.shape[:2]
        out = cv2.VideoWriter(..., (display_width, display_height))'''
        # 准备视频写入器
        if output_path:
            out = cv2.VideoWriter(output_path,
                                  cv2.VideoWriter_fourcc(*'XVID'),
                                  self.fps,
                                  (self.display_width, self.display_height))
        
        '''# 逐帧读取输入视频，直到视频结束或读取失败。
        ​关键变量：
        ret：布尔值，表示是否成功读取帧。
        frame：当前帧的BGR图像数据。'''
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_counter += 1
            
            '''使用背景减除算法（如MOG2/KNN）提取前景掩码，用于检测运动物体（导弹）。
            ​关键实现：
            self.bg_subtractor 需提前初始化（如 cv2.createBackgroundSubtractorMOG2()）。
            ​潜在问题：
            背景模型未充分学习导致前景噪声多。
            优化建议：
            # 前N帧用于训练背景模型
            if self.frame_counter < 30:
                self.bg_subtractor.apply(frame, learningRate=0.1)
            else:
                fg_mask = self.bg_subtractor.apply(frame)'''
            # 应用背景减除
            fg_mask = self.bg_subtractor.apply(frame)
            
            '''功能：基于前景掩码检测导弹位置，可能包括：
            形态学操作去噪。
            轮廓检测与过滤（按面积/长宽比）。
            更新导弹位置到 self.missile_bbox。
            ​潜在问题：
            复杂背景导致误检。
            ​优化建议：
            使用卡尔曼滤波或光流法跟踪导弹位置，提升稳定性。'''
            # 检测并跟踪导弹
            frame = self.detect_missile(frame, fg_mask)
            
            '''功能：判断导弹边界框（self.missile_bbox）是否与目标区域（self.target_roi）发生碰撞。
            ​关键逻辑：
            计算两个矩形的交集面积，若超过阈值则判定碰撞。
            碰撞后设置 self.impact_detected = True。
            ​优化建议：
            # 使用IoU（交并比）判断碰撞
            iou = calculate_iou(missile_bbox, target_roi)
            if iou > 0.1:
                self.impact_detected = True'''
            # 检测导弹与目标的接触
            frame = self.detect_impact(frame)
            
            '''功能：
            detect_explosion：通过亮度突变、颜色变化等检测爆炸发生。
            detect_explosion_end：通过多帧稳定性分析判断爆炸结束。
            ​关键参数：
            亮度阈值 self.brightness_threshold。
            稳定帧数阈值 self.stability_threshold。
            ​优化建议：
            使用光流法或能量变化模型提升爆炸检测准确性。'''
            # 检测爆炸
            frame = self.detect_explosion(frame)
            # 检测爆炸结束
            frame = self.detect_explosion_end(frame)
            
            # 创建显示帧
            display_frame = self.create_display_frame(frame)
            
            '''功能：实时显示处理结果并保存到视频文件。
            ​潜在问题：
            高性能场景下GUI显示延迟。
            ​优化建议：
            多线程分离显示和计算逻辑。'''
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
        
        '''功能：从摄像头获取帧率、分辨率等参数，并设置总帧数为无限（适用于实时流）。
        ​关键问题：
        部分摄像头返回的 fps 可能为0，导致后续计算错误（如 self.cooldown_frames）。
        ​优化建议：
        # 若摄像头未提供有效fps，设置默认值（如30）
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        # 或手动计算实际帧率
        start_time = time.time()
        for _ in range(30):
            ret, _ = self.cap.read()
        self.fps = 30.0 / (time.time() - start_time)'''
        # 获取摄像头的帧率和分辨率
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 由于摄像头是实时流，总帧数不适用
        self.total_frames = float('inf') # 实时流无固定帧数
        
        '''​功能：设置显示画面的宽度为原视频两倍（左右并排），初始化关键帧和时间戳字典。
        ​潜在问题：
        若摄像头分辨率过高（如4K），display_width 可能导致内存占用过大。
        ​优化建议：
        python
        # 限制显示分辨率，降低处理负载
        self.display_width = min(self.frame_width * 2, 1920)
        self.display_height = min(self.frame_height, 1080)'''
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
        
        '''功能：初始化背景减除器（MOG2算法），用于检测运动物体（导弹）。
        ​参数解析：
        history=500：考虑前500帧建模背景。
        varThreshold=16：像素与背景模型的马氏距离阈值，值越小灵敏度越高。
        detectShadows=False：不检测阴影，减少误检。
        ​潜在问题：
        固定参数可能不适应光照变化剧烈的场景。
        ​优化建议：
        # 动态调整背景减除参数
        self.bg_subtractor.setHistory(int(self.fps * 5))  # 5秒背景学习
        self.bg_subtractor.setVarThreshold(20 if lighting == 'low' else 10)'''
        self.target_roi = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        
        '''功能：配置光流法参数，用于跟踪导弹运动轨迹。
        ​关键参数：
        winSize=(15,15)：较大的窗口提高跟踪稳定性，但降低精度。
        maxLevel=2：使用两层图像金字塔，适应快速运动。
        ​潜在问题：
        高分辨率视频中角点数量不足可能导致跟踪失败。
        ​优化建议：
        python
        # 根据分辨率动态调整角点数量
        self.feature_params['maxCorners'] = min(200, (self.frame_width // 10) * (self.frame_height // 10))'''
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
        
        '''功能：初始化导弹检测和爆炸检测的阈值参数及状态标志。
        ​关键参数：
        explosion_min_area：爆炸区域至少占画面的1%，避免小区域误判。
        ​潜在问题：
        固定比例可能不适用于不同场景（如远距离小爆炸）。
        ​优化建议：
        python
        # 根据目标区域大小动态设置爆炸最小面积
        if self.target_roi:
            w, h = self.target_roi[2], self.target_roi[3]
            self.explosion_min_area = w * h * 0.5  # 目标区域的50%'''
        self.missile_bbox = None    # 导弹边界框
        self.explosion_threshold = 200 # 爆炸亮度阈值
        self.explosion_min_area = self.frame_width * self.frame_height * 0.01  # 爆炸最小区域
        # 状态标志
        self.missile_detected = False
        self.impact_detected = False
        self.explosion_detected = False
        self.explosion_end_detected = False
        
        '''功能：管理帧计数、碰撞冷却期、历史帧缓存等辅助参数。
        ​潜在问题：
        self.frames_to_keep=5 可能不足以捕捉快速变化。
        ​优化建议：
        # 根据帧率动态设置缓存大小
        self.frames_to_keep = max(5, int(self.fps * 0.5))  # 缓存0.5秒的帧'''
        self.frame_counter = 0            # 当前帧计数器
        self.impact_frame_count = 0       # 碰撞后冷却帧计数
        self.cooldown_frames = int(self.fps * 1.0)  # 碰撞后1秒内不重复检测
        self.previous_frames = []         # 缓存前N帧用于差分分析
        self.frames_to_keep = 5           # 缓存帧数
        self.brightness_threshold = 30
        self.color_change_threshold = 40
        self.stability_threshold = 30
        self.stability_counter = 0
        self.pixel_stability_threshold = 5
        self.previous_explosion_frames = []  # 爆炸帧缓存
        self.explosion_frames_to_keep = 10   # 爆炸分析缓存大小
        self.min_contour_area_ratio = 0.002  # 最小轮廓面积比
        
        '''功能：跨平台探测中文字体路径，确保中文文本正确渲染。
        ​潜在问题：
        路径硬编码可能导致Linux/macOS环境探测失败。
        未提供字体文件时的降级处理（如使用PIL默认字体）。
        ​优化建议：
        # 使用fonttools自动查找系统字体
        from fontTools import ttLib
        system_fonts = ttLib.get_system_fonts()
        for font in system_fonts:
            if "simhei" in font.lower() or "PingFang" in font:
                self.font_path = font
                break'''
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

    def process_video(self, output_path=None):
        """处理摄像头视频流并识别关键事件"""
        # 让用户选择目标区域
        self.select_target_roi()
        
        # 准备视频写入器
        if output_path:
            out = cv2.VideoWriter(output_path,
                                  cv2.VideoWriter_fourcc(*'XVID'),
                                  self.fps,
                                  (self.display_width, self.display_height))
        
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
            cv2.imshow("Missile Strike Analysis - Camera", display_frame)
            
            # 写入视频
            if output_path:
                out.write(display_frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 按'q'退出
                break
            elif key == ord('s') and output_path:  # 按's'保存当前帧
                frame_path = f"frame_{self.frame_counter}.jpg"
                cv2.imwrite(frame_path, display_frame)
                print(f"已保存当前帧到 {frame_path}")
        
        # 释放资源
        self.cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        return self.key_frames, self.key_frames_time
# def main():
#     parser = argparse.ArgumentParser(description='导弹打击目标视觉识别系统')
#     parser.add_argument('video_path', type=str, help='输入视频的路径')
#     parser.add_argument('--output', type=str, default=None, help='输出视频的路径')
#     args = parser.parse_args()
#     analyzer = MissileStrikeAnalyzer(args.video_path)
#     key_frames, key_frames_time = analyzer.process_video(args.output)

#     # 打印结果
#     print("\n检测到的关键事件:")
#     for event_name, time in key_frames_time.items():
#         event_display = {
#             "missile_appearance": "导弹出现",
#             "missile_impact": "导弹接触目标",
#             "explosion": "爆炸发生",
#             "explosion_end": "爆炸结束"  # 新增：爆炸结束显示
#         }
#         if time is not None:
#             print(f"{event_display[event_name]}: {analyzer.format_time(time)}")
#         else:
#             print(f"{event_display[event_name]}: 未检测到")

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

    # 根据参数选择初始化分析器
    if args.camera:
        analyzer = MissileStrikeAnalyzerCamera()  # 摄像头模式：实例化摄像头分析器类
    else:
        analyzer = MissileStrikeAnalyzer(args.video_path)  # 视频文件模式：传入视频路径实例化分析器

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
    '''关键代码解析及潜在优化点
    ​1. 命令行参数解析
    ​参数设计：
    --video_path: 默认视频文件为 Mk82.mp4，非必需参数。
    --camera: 布尔标志，启用摄像头模式。
    --output: 输出视频路径，可选。
    ​优化建议：
    ​互斥参数：使用 mutually_exclusive_group 确保 --video_path 和 --camera 不同时生效。
    python
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video_path', ...)
    group.add_argument('--camera', ...)
    ​2. 参数校验逻辑
    ​问题：当前校验仅检查 args.camera 和 args.video_path 是否同时为空，但未处理两者同时存在的情况。
    ​潜在风险：用户可能同时指定 --camera 和 --video_path，此时 args.camera 会覆盖视频路径。
    ​优化建议：
    python
    if args.camera and args.video_path:
        parser.error("--camera 和 --video_path 不能同时使用")
    ​3. 分析器初始化
    ​摄像头模式：MissileStrikeAnalyzerCamera 类需实现摄像头初始化和实时处理。
    ​视频模式：MissileStrikeAnalyzer 类需支持从视频文件读取帧。
    ​潜在问题：
    视频文件不存在时，MissileStrikeAnalyzer 可能抛出未处理的异常。
    ​优化建议：
    python
    try:
        analyzer = MissileStrikeAnalyzer(args.video_path)
    except FileNotFoundError:
        print(f"错误：视频文件 {args.video_path} 不存在！")
        sys.exit(1)
    ​4. 结果打印逻辑
    ​事件名称映射：将内部事件名（如 missile_appearance）转换为中文显示。
    ​时间格式化：analyzer.format_time(time) 将秒数转为 HH:MM:SS.ms 格式。
    ​优化建议：
    ​动态映射：若新增事件类型，需更新 event_display 字典。
    ​异常处理：处理 key_frames_time 中未知事件名的情况。
    python
    display_name = event_display.get(event_name, f"未知事件 ({event_name})")
    ​5. 资源管理
    ​摄像头/视频释放：analyzer.process_video() 需确保在退出前释放资源。
    ​潜在问题：若程序异常退出（如用户强制终止），资源可能未释放。
    ​优化建议：
    使用 try-finally 或上下文管理器确保资源释放：
    python
    try:
        key_frames, key_frames_time = analyzer.process_video(args.output)
    finally:
        analyzer.release_resources()  # 自定义释放方法'''

if __name__ == "__main__":
    main()  # 程序入口