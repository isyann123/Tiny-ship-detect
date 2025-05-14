#### 新增代码部分
在loss.py文件夹中
`class BboxLoss(nn.Module):`类 内部
将
```python
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
```
替换成
```python
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        b1_x1, b1_y1, b1_x2, b1_y2 = pred_bboxes[fg_mask].chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = target_bboxes[fg_mask].chunk(4, -1)
        BX_L2Norm = torch.pow((b1_x1 - b2_x1), 2)
        BY_L2Norm = torch.pow((b1_y1 - b2_y1), 2)
        p1 = BX_L2Norm + BY_L2Norm
        w_FroNorm = torch.pow((b1_x2 - b2_x2)/2, 2)
        h_FroNorm = torch.pow((b1_y2 - b2_y2)/2, 2)
        p2 = w_FroNorm + h_FroNorm
        wasserstein = torch.exp(-torch.pow((p1+p2), 1 / 2) / 2.5)
        wdloss = True # 设置为 True 使用Normalized Gaussian Wasserstein Distance 设置 False 则用v8默认的
        if wdloss:
            loss_iou = (0.7 * ((1.0 - iou) * weight).sum() + 0.3 * ((1.0 - wasserstein) * weight).sum()) / target_scores_sum
        else:
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
```