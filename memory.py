import torch
from torchstat import stat
from ghostnet import ghostnet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "weights/model-9.pth"
model = ghostnet(input_channel=3, num_classes=6)
model.load_state_dict(torch.load(model_path, map_location=device))
# 导入模型，输入一张输入图片的尺寸
stat(model, (3, 224, 224))
