import os
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

# 配置路径
dataset_path = Path("E:\FAIR-CSAR")
output_path = Path("E:\FAIR-CSAR-ship_only")
target_classes = {
    "Container_Ship",
    "General_Cargo_Ship",
    "Motion_Defocusing_Ship",
    "Warship",
    "Other_Ship"
}


def process_split(split: str):
    """处理单个数据集分割（train/test）"""
    # 创建输出目录
    (output_path / split / "PNGImages").mkdir(parents=True, exist_ok=True)
    (output_path / split / "Annotations").mkdir(parents=True, exist_ok=True)
    (output_path / split / "KeyPoints").mkdir(parents=True, exist_ok=True)

    # 记录有效文件基名（不含后缀）
    valid_basenames = set()

    # ===== 第一步：筛选DOTA标签 =====
    dota_dir = dataset_path / split / "Annotations"
    for txt_file in dota_dir.glob("*.txt"):
        with open(txt_file) as f:
            lines = [line.strip() for line in f
                     if line.strip() and not line.startswith('imagesize')]

        # 筛选包含目标类别的标注行
        selected_lines = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 9 and parts[8] in target_classes:
                selected_lines.append(line)

        # 保存有效文件
        if selected_lines:
            valid_basenames.add(txt_file.stem)
            with open(output_path / split / "Annotations" / txt_file.name, 'w') as f:
                f.write("\n".join(selected_lines))

    # ===== 第二步：同步筛选关键点和图像 =====
    for basename in valid_basenames:
        # 处理关键点XML
        xml_src = dataset_path / split / "KeyPoints" / f"{basename}.xml"
        if xml_src.exists():
            try:
                tree = ET.parse(xml_src)
                root = tree.getroot()

                # 筛选目标类别对象
                for obj in root.findall('object'):
                    if obj.find('name').text not in target_classes:
                        root.remove(obj)

                # 只保留包含有效对象的XML
                if root.find('object') is not None:
                    # 保持XML格式规范
                    tree.write(output_path / split / "KeyPoints" / f"{basename}.xml",
                               encoding='utf-8', xml_declaration=True)

                    # 复制对应图像（自动匹配扩展名）
                    for ext in ['.png', '.jpg', '.tif', '.bmp']:
                        img_src = dataset_path / split / "PNGImages" / f"{basename}{ext}"
                        if img_src.exists():
                            shutil.copy2(img_src, output_path / split / "PNGImages" / img_src.name)
                            break
            except ET.ParseError:
                print(f"XML解析失败：{xml_src}")


# 执行处理
for split in ["train", "test"]:
    process_split(split)
print("筛选完成！结果保存在：", output_path)