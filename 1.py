import os
import xml.etree.ElementTree as ET
import traceback

ANNOTATION_DIR = r"E:\FAIR-CSAR-ship_only\test\Annotations"
KEYPOINT_DIR    = r"E:\FAIR-CSAR-ship_only\test\KeyPoints"
OUTPUT_DIR      = r"E:\FAIR-CSAR-ship_only\test\Labels"
IMAGE_WIDTH     = 1024
IMAGE_HEIGHT    = 1024

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_bbox(line, file, line_num):
    try:
        parts = line.strip().split()
        if len(parts) < 9:
            raise ValueError(f"字段不足，预计至少9列，实际{len(parts)}列")
        coords = list(map(float, parts[:8]))
        class_id = parts[-1]
        x_pts = coords[0::2]; y_pts = coords[1::2]
        x_min, x_max = min(x_pts), max(x_pts)
        y_min, y_max = min(y_pts), max(y_pts)
        xc = ((x_min + x_max) / 2) / IMAGE_WIDTH
        yc = ((y_min + y_max) / 2) / IMAGE_HEIGHT
        w  = (x_max - x_min)   / IMAGE_WIDTH
        h  = (y_max - y_min)   / IMAGE_HEIGHT
        return class_id, [xc, yc, w, h]
    except Exception as e:
        print(f"[ERROR] {file} 第 {line_num} 行解析边框失败：{e}")
        traceback.print_exc()
        return None, None

def parse_keypoints(xml_path, file):
    kps_all = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[ERROR] 解析 XML 文件失败：{xml_path}，{e}")
        traceback.print_exc()
        return None

    for idx, obj in enumerate(root.findall("object"), start=1):
        try:
            keypoints_elem = obj.find("keypoints")
            if keypoints_elem is None:
                print(f"[WARNING] {file} 第 {idx} 个 object 缺少 <keypoints> 节点")
                kps_all.append([])
                continue

            # 提取所有 x1~x9 和 y1~y9
            kps = []
            for i in range(1, 10):  # x1~x9, y1~y9
                x_tag = keypoints_elem.find(f"x{i}")
                y_tag = keypoints_elem.find(f"y{i}")
                if x_tag is None or y_tag is None:
                    print(f"[WARNING] {file} object {idx} 缺少 x{i} 或 y{i}，跳过该点")
                    continue
                try:
                    x = float(x_tag.text) / IMAGE_WIDTH
                    y = float(y_tag.text) / IMAGE_HEIGHT
                    v = 2  # 没有 visible 字段，我们设为“标注但不评估”
                    kps.extend([x, y, v])
                except Exception as e:
                    print(f"[ERROR] {file} object {idx} 点 {i} 坐标格式错误：{e}")
            kps_all.append(kps)
        except Exception as e:
            print(f"[ERROR] {file} 中第 {idx} 个 object 提取关键点失败：{e}")
            traceback.print_exc()
            return None

    return kps_all

# 主处理循环
for fname in os.listdir(ANNOTATION_DIR):
    if not fname.endswith(".txt"):
        continue

    base = os.path.splitext(fname)[0]
    ann_path = os.path.join(ANNOTATION_DIR, fname)
    kp_path  = os.path.join(KEYPOINT_DIR,    base + ".xml")
    out_path = os.path.join(OUTPUT_DIR,      base + ".txt")

    if not os.path.isfile(kp_path):
        print(f"[WARNING] 找不到对应关键点文件：{kp_path}")
        continue

    # 读取标注行
    try:
        with open(ann_path, 'r', encoding='utf-8') as f:
            ann_lines = f.readlines()
    except Exception as e:
        print(f"[ERROR] 打开标注文件失败：{ann_path}，{e}")
        continue

    # 解析关键点
    keypoints_all = parse_keypoints(kp_path, kp_path)
    if keypoints_all is None:
        continue

    # 数量一致性检查
    if len(ann_lines) != len(keypoints_all):
        print(f"[ERROR] 对象数不一致：{fname} 中有 {len(ann_lines)} 个 bbox，"
              f"但 {base}.xml 中有 {len(keypoints_all)} 个 object")
        continue

    # 写入输出
    with open(out_path, 'w', encoding='utf-8') as out_f:
        for idx, (line, kps) in enumerate(zip(ann_lines, keypoints_all), start=1):
            cid, bbox = parse_bbox(line, ann_path, idx)
            if bbox is None:
                # 如果解析失败，跳过该行
                continue
            try:
                vals = bbox + kps
                out_f.write(f"{cid} " + " ".join(f"{v:.6f}" for v in vals) + "\n")
            except Exception as e:
                print(f"[ERROR] 写入 {out_path} 时，第 {idx} 行合并失败：{e}")
                traceback.print_exc()

print("转换完成，详情请查看上述日志信息。")
