import os
import json

def filter_ships_to_yolo(json_path, image_dir, output_json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 获取类别映射
    id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    name_to_id = {v: k for k, v in id_to_name.items()}

    # 获取船只类别 ID
    target_id = name_to_id.get('ship')
    if target_id is None:
        print("未找到船只类别")
        return

    print(f"筛选类别：{id_to_name[target_id]} (ID={target_id})")

    # 图像 ID -> 图像信息
    images = {img['id']: img for img in data['images']}
    missing_images = set()  # 使用集合避免重复的图片文件名

    for ann in data['annotations']:
        if ann['category_id'] != target_id:
            continue

        image_id = ann['image_id']
        img_info = images.get(image_id)

        if img_info is None:
            continue  # 如果图像信息缺失，跳过

        img_name = img_info['file_name']
        img_path = os.path.join(image_dir, img_name)

        # 如果图片不存在，记录文件名
        if not os.path.exists(img_path):
            missing_images.add(img_name)  # 使用集合来避免重复

    # 保存未找到的图片文件名到 JSON 文件
    if missing_images:
        with open(output_json_path, 'w') as output_file:
            json.dump(list(missing_images), output_file, indent=4)  # 转换为列表
        print(f"未找到的图片文件已保存至 {output_json_path}")
    else:
        print("所有图片均已找到")


# 示例调用
filter_ships_to_yolo(
    json_path='E:/aitodv2_test.json',  # 标注文件路径
    image_dir='E:/AITOD/images',                   # 图片文件夹路径
    output_json_path='E:/AITOD-ship_only/test/missing_images.json'         # 输出未找到图片文件的 JSON 文件路径
)
