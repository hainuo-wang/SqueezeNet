import torch
from torchstat import stat
from model import SqueezeNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "weights/model-149.pth"
model = SqueezeNet(in_channel=1, num_classes=7)
model.load_state_dict(torch.load(model_path, map_location=device))
# 导入模型，输入一张输入图片的尺寸
stat(model, (1, 224, 224))
