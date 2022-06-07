import torch
print(torch.__version__)
#state_dict = torch.load('/home/liyang/YOLOX/YOLOX_M/yolox_ahs_v3_0.9806/best_ckpt.pth')
#torch.save(state_dict, '/home/liyang/YOLOX/YOLOX_M/yolox_ahs_v3_0.9806/converted_best_ckpt.pth', _use_new_zipfile_serialization=False)
#state_dict = torch.load('/home/liyang/YOLOX/YOLOX_tiny_without_focus_v2/latest_ckpt.pth')
#torch.save(state_dict, '/home/liyang/YOLOX/YOLOX_tiny_without_focus_v2/torch1.5.0_latest_ckpt.pth', _use_new_zipfile_serialization=False)
state_dict = torch.load('/home/liyang/docker_yolox/yolox_s.pth')
torch.save(state_dict, '/home/liyang/docker_yolox/new_yolox_s.pth', _use_new_zipfile_serialization=False)
