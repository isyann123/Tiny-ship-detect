import cv2


def txtShow(img,txt,save=True):
    image = cv2.imread(img)
    height,width = image.shape[:2]      # 获取原始图像的高和宽

    # 读取classes类别信息
    with open('data/images/classes.txt', 'r') as f:  # 获取类别
        classes = f.read().splitlines()

    # 读取yolo格式标注的txt信息
    with open(txt,'r') as f:
        labels = f.read().splitlines()

    ob = []         # 存放目标信息
    for i in labels:
        cl, x_centre, y_centre, w, h = i.split(' ')

        # 需要将数据类型转换成数字型
        cl, x_centre, y_centre, w, h = int(cl), float(x_centre), float(y_centre), float(w),float(h)
        name = classes[cl]      # 根据classes文件获取真实目标
        xmin = int(x_centre * width - w * width / 2)        # 坐标转换
        ymin = int(y_centre * height - h * height / 2)
        xmax = int(x_centre * width + w * width / 2)
        ymax = int(y_centre * height + h * height / 2)

        tmp = [name, xmin, ymin, xmax, ymax]  # 单个检测框
        ob.append(tmp)

    # 绘制检测框
    for name, x1, y1, x2, y2 in ob:
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=3)  # 绘制矩形框
        cv2.putText(image, name, (x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, thickness=2, color=(0, 255, 0))


        # 展示图像
    cv2.imshow('test', image)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__=='__main__':
    img_path = 'data/images/Clownfish1.jpg'      # 传入图片
    label_path = 'data/images/Clownfish1.txt'    # 传入标签
    txtShow(img=img_path,txt=label_path,save=True)