import base64
import requests
import json

def classify_scene(api_key, image_path):
    """
    首先对图片进行场景分类，判断属于哪个场景
    
    Args:
        api_key: 阿里云DashScope API密钥
        image_path: 图片路径
        
    Returns:
        场景类型: "导弹出现", "导弹接触目标", "爆炸发生", "爆炸结束"
    """
    # 读取并编码图片
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        
    # 构建请求
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 场景分类的提示词
    classification_prompt = """请分析这张图片，判断它属于以下哪个场景类型，只需要回答场景类型名称：
1. 导弹出现 - 图像中可以看到导弹（通常在图像边缘有绿色框选），但还未接触目标
2. 导弹接触目标 - 导弹正在或刚刚接触到目标
3. 爆炸发生 - 可以看到明显的爆炸火光、火球或爆炸效果
4. 爆炸结束 - 爆炸已经结束，主要是烟雾、碎片或爆炸后的场景

请只回答：导弹出现、导弹接触目标、爆炸发生、爆炸结束 中的一个。"""
    
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
                            "text": classification_prompt
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
            
            # 处理返回格式
            if isinstance(content, list) and len(content) > 0:
                if isinstance(content[0], dict) and 'text' in content[0]:
                    scene_type = content[0]['text'].strip()
                else:
                    scene_type = str(content[0]).strip()
            else:
                scene_type = str(content).strip()
            
            # 确保返回的是预期的场景类型之一
            valid_scenes = ["导弹出现", "导弹接触目标", "爆炸发生", "爆炸结束"]
            for scene in valid_scenes:
                if scene in scene_type:
                    return scene
            
            # 如果没有匹配到，默认返回导弹出现
            return "导弹出现"
        else:
            print(f"场景分类API错误 {response.status_code}: {response.text}")
            return "导弹出现"  # 默认场景
            
    except Exception as e:
        print(f"场景分类请求失败: {str(e)}")
        return "导弹出现"  # 默认场景

def get_scene_prompt(scene_type):
    """
    根据场景类型返回对应的提示词
    
    Args:
        scene_type: 场景类型
        
    Returns:
        对应的提示词
    """
    prompts = {
        "导弹出现": """这是一个{导弹出现}的场景，其中的导弹是位于图像边缘的绿色框选对象，你需要对图像进行理解和分析，并严格按照以下JSON格式输出五元组结果，不要添加任何其他内容：

{
  "event_type": "场景类型",
  "pentuple": {
    "primary_subject": "主体对象（参与事件的核心能动实体）",
    "target_object": "客体/目标（事件的主要承受者或作用对象）", 
    "key_action": "关键行为/事件（主体与客体之间发生的核心交互）",
    "effect_characterization": "效能表征细节（事件结果和过程的关键视觉细节）",
    "spatio_temporal_context": {
      "timestamp_sec": [时间戳],
      "impact_bbox_px": [x_min, y_min, x_max, y_max]
    }
  }
}""",
        
        "导弹接触目标": """这是一个{导弹接触目标}的场景，你需要对图像进行理解和分析，并严格按照以下JSON格式输出五元组结果，不要添加任何其他内容：

{
  "event_type": "场景类型",
  "pentuple": {
    "primary_subject": "主体对象（参与事件的核心能动实体）",
    "target_object": "客体/目标（事件的主要承受者或作用对象）", 
    "key_action": "关键行为/事件（主体与客体之间发生的核心交互）",
    "effect_characterization": "效能表征细节（事件结果和过程的关键视觉细节）",
    "spatio_temporal_context": {
      "timestamp_sec": [时间戳],
      "impact_bbox_px": [x_min, y_min, x_max, y_max]
    }
  }
}""",
        
        "爆炸发生": """这是一个{爆炸发生}的场景，你需要对图像进行理解和分析，并严格按照以下JSON格式输出五元组结果，不要添加任何其他内容：

{
  "event_type": "场景类型",
  "pentuple": {
    "primary_subject": "主体对象（参与事件的核心能动实体）",
    "target_object": "客体/目标（事件的主要承受者或作用对象）", 
    "key_action": "关键行为/事件（主体与客体之间发生的核心交互）",
    "effect_characterization": "效能表征细节（事件结果和过程的关键视觉细节）",
    "spatio_temporal_context": {
      "timestamp_sec": [时间戳],
      "impact_bbox_px": [x_min, y_min, x_max, y_max]
    }
  }
}""",
        
        "爆炸结束": """这是一个{爆炸结束}的场景，你需要对图像进行理解和分析，并严格按照以下JSON格式输出五元组结果，不要添加任何其他内容：

{
  "event_type": "场景类型",
  "pentuple": {
    "primary_subject": "主体对象（参与事件的核心能动实体）",
    "target_object": "客体/目标（事件的主要承受者或作用对象）", 
    "key_action": "关键行为/事件（主体与客体之间发生的核心交互）",
    "effect_characterization": "效能表征细节（事件结果和过程的关键视觉细节）",
    "spatio_temporal_context": {
      "timestamp_sec": [时间戳],
      "impact_bbox_px": [x_min, y_min, x_max, y_max]
    }
  }
}"""
    }
    
    return prompts.get(scene_type, prompts["导弹出现"])

def analyze_image(api_key, image_path, prompt):
    """
    使用阿里云千问VL分析图片
    
    Args:
        api_key: 阿里云DashScope API密钥
        image_path: 图片路径
        prompt: 用户输入的提示词
        
    Returns:
        分析结果文本
    """
    # 读取并编码图片
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        
    # 构建请求
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "qwen-vl-plus",  # 可选: qwen-vl-plus, qwen-vl-max
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
            
            # 处理返回格式：如果是列表格式，提取文本内容
            if isinstance(content, list) and len(content) > 0:
                if isinstance(content[0], dict) and 'text' in content[0]:
                    return content[0]['text']
                else:
                    return str(content[0])
            else:
                return str(content)
        else:
            return f"API错误 {response.status_code}: {response.text}"
            
    except requests.exceptions.Timeout:
        return "请求超时，请重试"
    except Exception as e:
        return f"请求失败: {str(e)}"

def main():
    """主程序"""
    
    # 输入API密钥
    api_key = "sk-2d64c5ef5b434265b535122a12aa9cea"  # 替换为你的API密钥
    
    while True:
        print("\n" + "="*40)
        
        # 输入图片路径
        # image_path = "C:/Users/13427/Desktop/API_diaoyong/1.png"  # 替换为你的图片路径
        image_path = input("请输入图片路径（或输入'quit'退出）：").strip()    # 手动输入图片路径
        
        if image_path.lower() == 'quit':
            print("再见!")
            break
        
        try:
            # 第一步：自动识别场景类型
            print("\n正在识别场景类型...")
            scene_type = classify_scene(api_key, image_path)
            print(f"识别的场景类型: {scene_type}")
            
            # 第二步：根据场景类型获取对应的提示词
            prompt = get_scene_prompt(scene_type)
            print(f"使用的提示词: {prompt}")
            
            # 第三步：分析图片并获取五元组
            print("\n正在分析图片并生成五元组...")
            result = analyze_image(api_key, image_path, prompt)
            print("\n分析结果:")
            print("-" * 40)
            print(result)
            print("-" * 40)
            
        except FileNotFoundError:
            print(f"找不到图片文件: {image_path}")
        except Exception as e:
            print(f"分析失败: {str(e)}")
        
        # 询问是否继续分析其他图片
        continue_choice = input("\n是否继续分析其他图片？(y/n): ").strip().lower()
        if continue_choice != 'y':
            print("再见!")
            break

if __name__ == "__main__":
    main()