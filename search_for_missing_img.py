import json

def count_annotated_images(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    image_count = len(data.get("images", []))
    print(f"该 JSON 文件中标注的图片总数为：{image_count}")

# 示例调用
count_annotated_images("E:/aitodv2_test.json")  # 替换为你的 JSON 文件路径
