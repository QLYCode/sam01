import numpy as np
import torch
from skimage import io, transform
import cv2
import logging

from networks.MedSAM_Inference import medsam_model, medsam_inference

import utils.helpers as helpers

def get_sam_seg(num_classes, outputs, labels, processed_images, shape, writer, iter_num):
    B, C, H, W = shape
    outputs_np = torch.argmax(outputs, dim=1).cpu().numpy()  # Shape: (B, H, W)
    bounding_boxes = []
    edges_list = []
    medsam_seg_list = []
    for class_id in range(num_classes):  # 遍历每个类别
        medsam_seg_batch_list = []
        for batch_idx in range(outputs_np.shape[0]):  # 遍历 batch 中的每个样本
            class_mask = (outputs_np[batch_idx] == class_id).astype(np.uint8)  # 类别 mask

            if class_mask.any():  # 如果类别 mask 不为空
                # 提取 bounding box
                contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                x, y, w, h = cv2.boundingRect(contours[0])
                bounding_boxes.append([batch_idx, class_id, x, y, w, h])

                # 边缘检测
                edges = cv2.Canny(class_mask * 255, 100, 200)
                edges_list.append((batch_idx, class_id, edges))

                # 记录到 TensorBoard
                box_np = np.array([x, y, x + w, y + h])
                box_1024 = box_np / np.array([W, H, W, H]) * 1024  # 转换到 1024x1024 尺度

                # 使用 med-sam 生成分割结果
                with torch.no_grad():
                    image_embedding = medsam_model.image_encoder(processed_images[batch_idx])  
                    medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024.reshape(1,4), H, W)

                # 将结果记录到 TensorBoard
                medsam_seg_tensor = torch.tensor(medsam_seg).unsqueeze(0).float().to(labels.device)  # (1, H, W)
                medsam_seg_batch_list.append(medsam_seg_tensor)
                if class_id == 0 and batch_idx == 0 and iter_num % 20 == 0:
                    writer.add_image(f'train/medsam_seg_class_{class_id}_batch_{batch_idx}', medsam_seg_tensor,
                                        iter_num)

                # print(f"Batch {batch_idx}, Class {class_id}: Bounding Box (1024x1024 scale): {box_1024}")
            else:
                # 如果类别 mask 为空，添加一个全零张量作为占位符
                empty_seg_tensor = torch.zeros((1, H, W), dtype=torch.float).to(labels.device)
                medsam_seg_batch_list.append(empty_seg_tensor)
        medsam_seg_list.append(medsam_seg_batch_list)
    return medsam_seg_list, edges_list

def train_epoch(data_loader, model, args, writer, loss, optimizer, iter_num, device):
    model.train()
    for sampled_batch in data_loader:
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        shape = volume_batch.shape
        processed_images = helpers.process_images(volume_batch, device)
        volume_batch = volume_batch.to(device)
        label_batch = label_batch.to(device)
        
        helpers.check_for_nan(volume_batch)
        outputs = model(volume_batch)  # generate coarse prediction
        outputs_soft = torch.softmax(outputs, dim=1)

        medsam_seg_list, edges_list = get_sam_seg(args.num_classes, outputs_soft, label_batch, processed_images, shape, writer, iter_num)
        loss_ce = loss(outputs, label_batch[:].long())
        medsam_seg_output = torch.cat([torch.cat(medsam_seg_class_list, dim=0) for medsam_seg_class_list in medsam_seg_list], dim=0).view(args.batch_size, args.num_classes, 256, 256).to(label_batch.device)
        medsam_seg_loss = loss(medsam_seg_output, label_batch[:].long())

        total_loss = loss_ce + medsam_seg_loss
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        lr_ = args.base_lr * (1.0 - iter_num / args.max_iterations) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
        scalars = {
            "lr": lr_,
            "total_loss": total_loss,
            "loss_ce": loss_ce,
            "medsam_seg_loss": medsam_seg_loss
        }
        helpers.log_epoch("train", writer, iter_num, scalars, volume_batch, outputs_soft, label_batch, edges_list)

        iter_num += 1
        

        # if iter_num >= args.max_iterations:
        if iter_num > 25:
            break
    return iter_num
        