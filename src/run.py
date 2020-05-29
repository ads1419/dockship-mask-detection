#!/usr/bin/env python
# coding: utf-8


import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode
from fastai.vision import *
from pathlib import Path
import PIL
import torchvision.transforms as T
from tqdm import tqdm
from utils.timer import Timer
import argparse


parser = argparse.ArgumentParser(description='MaskDetection')
parser.add_argument('--md', default='./weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--mc', default='./weights/trained-resnet34.pkl',
                    type=str, help='Trained state_dict file path to open')

parser.add_argument('--input', default='../input/', type=str, help='Dir to fetch input images')
parser.add_argument('--output', default='../output/', type=str, help='Dir to save image results')


parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')

parser.add_argument('--thresh', default=0.75, type=float, help='Confidence threshold')
parser.add_argument('--resize', default=0.75, type=float, help='Scaling factor for testing images')
parser.add_argument('--save_image', default=True, help='Save images to output folder')
parser.add_argument('--save_dets', default=True, help='Save detections as text files')

args = parser.parse_args()


# In[8]:


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


# In[9]:


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


# In[10]:


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def get_face_bboxes():
    good_dets = {}
    for image in tqdm(os.listdir(input_dir)):
        img_raw = cv2.imread(str(input_dir/image), cv2.IMREAD_COLOR)
        img = np.float32(img_raw)
        
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
            
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        _t[forward_pass_det].tic() 
        loc, conf, _ = net(img)  # forward pass
        _t[forward_pass_det].toc()
        
        _t[misc_det].tic()
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        # landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        # landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        # landms = landms[keep]

        good_det = []
        for b in dets:
            if b[4] < vis_thres:
                continue
            good_det.append(b)

        good_dets[image] = good_det
        _t[misc_det].toc()

    print('im_detect: [forward_pass_time: {:.4f}s misc: {:.4f}s] per image'.format(_t[forward_pass_det].average_time, _t[misc_det].average_time))
    return good_dets


# In[11]:


input_dir = Path(args.input)
output_dir = Path(args.output)


torch.set_grad_enabled(False)
cfg = cfg_mnet
cpu = args.cpu
origin_size = True
confidence_threshold = 0.02
nms_threshold = 0.4
save_folder = output_dir

vis_thres = args.thresh
resize = args.resize



if __name__ == '__main__':


    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.md, cpu)
    net.eval()
    print('Finished loading detection model!')
    # print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if cpu else "cuda")
    net = net.to(device)


    # In[18]:


    forward_pass_det = 'forward_pass_det'
    forward_pass_cls = 'forward_pass_cls'
    misc_det = 'misc_det'
    misc_cls = 'misc_cls'


    # In[19]:


    _t = {forward_pass_det: Timer(), forward_pass_cls: Timer(), misc_det: Timer(), misc_cls: Timer()}




    # ## Classifier

    # In[20]:


    model_path = Path(args.mc)
    learn = load_learner(model_path, "")
    print(f"\nLoading pretrained model from {str(args.mc)}")
    print('Finished loading detection model!')

    # In[21]:


    print("\n\nGETTING DETECTIONS...")
    good_dets = get_face_bboxes()


    # In[22]:


    print("\n\nGETTING CLASSIFICATIONS...")


    # In[24]:

    if not os.path.exists(output_dir): os.makedirs(output_dir)

    det_path = output_dir/"detections"
    if args.save_dets: 
        if not os.path.exists(det_path): os.makedirs(det_path)


    for image in tqdm(list(good_dets.keys())):

        img = PIL.Image.open(input_dir/image)
        w, h = img.size
        preds = []

    #     if(len(good_dets[image]) < 1):
    #         continue

        for box in good_dets[image]:
            scale = 0.5

            xpixels = int((box[2] - box[0]) * scale)
            ypixels = int((box[3] - box[1]) * scale)

            xmin = max(box[0] - xpixels, 0.)
            ymin = max(box[1] - ypixels, 0.)
            xmax = min(box[2] + xpixels, w)
            ymax = min(box[3] + ypixels, h)

            crop_img = img.crop((xmin, ymin, xmax, ymax))
            new_img = crop_img.resize((224,224))

            img_tensor = T.ToTensor()((new_img).convert("RGB"))
            img_fastai = Image(img_tensor)
            try:
                _t[forward_pass_cls].tic()
                preds.append(learn.predict(img_fastai))
                _t[forward_pass_cls].toc()
            except:
                print(image, img.size, (xmin, ymin, xmax, ymax))

        _t[misc_cls].tic()
        cats = ["face", "face_mask"]
        img_raw = cv2.imread(str(input_dir/image), cv2.IMREAD_COLOR)

        if args.save_dets: 
            name = os.path.splitext(Path(image))[0] + '.txt'
            f = open(det_path/name, "w+")

        for i, b in enumerate(good_dets[image]):
            if int(preds[i][0]): color = (0, 255, 0)
            else: color = (0, 0, 255)

            if args.save_dets: f.write(f"{cats[int(preds[i][0])]} {b[4]} {int(b[0])} {int(b[1])} {int(b[2])} {int(b[3])}\n")

            if args.save_image:
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), color, 2);
                cx = int(b[0])
                cy = int(b[1] + 12)
                text = f"{cats[int(preds[i][0])]}" + " {:.4f}".format(b[4])
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))


        if args.save_dets:
            f.close()

        if args.save_image:
            name = str(output_dir/image)
            cv2.imwrite(name, img_raw)

        _t[misc_cls].toc()


    print(f"im_classify: [forward_pass_time: {_t[forward_pass_cls].average_time:.4f}s misc: {_t[misc_cls].average_time:.4f}s] per image\n\n")
    if args.save_image: print(f"IMAGES SAVED in {output_dir}")
    if args.save_dets: print(f"DETECTIONS SAVED in {str(det_path)}\n\n")




