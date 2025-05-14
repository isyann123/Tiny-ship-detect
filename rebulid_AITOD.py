import os
import json
import shutil

def filter_ships_to_yolo(json_path, image_dir, output_image_dir, output_label_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

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
    label_map = {}

    for ann in data['annotations']:
        if ann['category_id'] != target_id:
            continue

        image_id = ann['image_id']
        if image_id not in label_map:
            label_map[image_id] = []

        img_info = images[image_id]
        width = img_info['width']
        height = img_info['height']

        x, y, w, h = ann['bbox']
        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        w /= width
        h /= height

        label_line = f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
        label_map[image_id].append(label_line)

    # 写入标签并复制图像
    for image_id, labels in label_map.items():
        img_info = images[image_id]
        img_name = img_info['file_name']
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            print(f"图片不存在: {img_path}")
            continue

        # 写 label 文件
        label_file = os.path.splitext(img_name)[0] + ".txt"
        with open(os.path.join(output_label_dir, label_file), 'w') as f:
            f.write('\n'.join(labels))

        # 复制图像
        shutil.copy(img_path, os.path.join(output_image_dir, img_name))

    print(f"共生成 {len(label_map)} 张图像与对应的标签")

# 示例调用
filter_ships_to_yolo(
    json_path='E:/AITOD/complete_annotations/aitod_test_v1_1.0.json',  # 标注文件路径
    image_dir='E:/AITOD/images',                   # 图片文件夹路径
    output_image_dir='E:/AITODv2-ship_only/test/images',       # 输出图片目录
    output_label_dir='E:/AITODv2-ship_only/test/labels'        # 输出标签目录
)
