import torch
import cv2
import time
import numpy as np
import torchvision
from yolox.utils import postprocess


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes

    # Settings
    # (pixels) minimum and maximum box width and height
    max_wh = 4096
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 1.0  # seconds to quit after
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[x[..., 4] > conf_thres]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        #x[:, 5] *= x[:, 4] 

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
           # i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            i, j =((x[:, 5:] > conf_thres).nonzero(as_tuple=False)).cpu().numpy().T
            x = torch.cat((box[i],x[i, j + 5, None].clone().detach().cuda(),torch.from_numpy(j[:, None]).clone().detach().cuda().float()), 1)
            #x = torch.cat((box[i],x[i, j + 5, None].clone().detach(),torch.from_numpy(j[:, None]).clone().detach().float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        import torchvision
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]

        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    try:
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    except cv2.error as e:
        print(e)
        pass
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

if __name__ == '__main__':
    path = '/home/liyang/YOLOX/yolox.torchscript.pt'
    model = torch.jit.load(path)
    model.eval()
    nms = 0.5
    conf = 0.6
    num_classes = 3
    input_shape = (416, 416)
    cls_name = ['face', 'logo', 'badge']
    img_path = '/world/data-gpu-94/liyang/aihuishou_train/yolox_ahs_dataset/test.json'
    with open(img_path, 'r') as f:
        lines = f.readlines()
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for i, line in enumerate(lines):
            # box: list contains[cls_name, score, x0, y0 ,x1 ,y1]
            box = []
            line = line.strip().split('\t')
            #get img absolute path
            k = line[0] 
            image = cv2.imread(k)
            #preprocess
            img, ratio = preproc(image, input_shape)
            #transpose
            img = torch.unsqueeze(torch.from_numpy(img), 0)
            img = img.float().cuda()
            #forward
            outputs = model(img)
            #postprocess
            #outputs = postprocess(outputs, num_classes, conf, nms, class_agnostic=True)
            outputs = non_max_suppression(outputs, conf, nms)
            if outputs is None or outputs[0] is None:
                continue
            for k in range(outputs[0].shape[0]):
                box_dic = {}
                x0 = outputs[0][k, 0].item() /ratio
                y0 = outputs[0][k, 1].item() /ratio
                x1 = outputs[0][k, 2].item() /ratio
                y1 = outputs[0][k, 3].item() /ratio
                obj_conf = outputs[0][k, 4].item()
                #cls_conf = outputs[0][k, 5].item()
                cls = outputs[0][k, 5].item()
                cls = int(cls)
                if cls == 0:
                    cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), color[cls])
                if cls == 1:
                    cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), color[cls])
                if cls == 2:
                    cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), color[cls])
                #box.append([cls_name[cls], obj_conf, x0, y0, x1, y1])
                
            #print detection results of one image and exit    
            #cv2.imwrite('/world/data-gpu-94/liyang/pedDetection/Faceboxes_train_samples/{}.jpg'.format(i), image)
            #print(box)    
            #exit()
