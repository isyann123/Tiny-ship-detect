import os
import json

def coco_to_yolo(json_path, output_folder):
    with open(json_path, 'r') as f:
        data = json.load(f)

    images = {img['id']: img for img in data['images']}
    categories = {cat['id']: idx for idx, cat in enumerate(data['categories'])}

    os.makedirs(output_folder, exist_ok=True)

    for ann in data['annotations']:
        image_id = ann['image_id']
        image_info = images[image_id]
        image_name = os.path.splitext(image_info['file_name'])[0] + '.txt'
        width = image_info['width']
        height = image_info['height']

        x, y, w, h = ann['bbox']
        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        w /= width
        h /= height

        category_id = ann['category_id']
        class_id = categories[category_id]  # Convert COCO id to 0-based YOLO id

        line = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

        txt_path = os.path.join(output_folder, image_name)
        with open(txt_path, 'a') as out_file:
            out_file.write(line + '\n')

    print(f"✅ 转换完成！YOLO格式标注保存在 {output_folder}/")

coco_to_yolo(
    json_path='E:/AITOD/complete_annotations/aitod_test_v1_1.0.json',
    output_folder='E:/AITOD/test_labels'
)
