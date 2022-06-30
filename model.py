# 以下を「model.py」に書き込み
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

classes_ja = ["ステージ_01", "ステージ_02", "ステージ_03"]
classes_en = ['stage_01', 'stage_02', 'stage_03']
n_class = len(classes_ja)
img_size = 32

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 畳み込み層:(入力チャンネル数, フィルタ数、フィルタサイズ)
        self.pool = nn.MaxPool2d(2, 2)  # プーリング層:（領域のサイズ, ストライド）
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 256)  # 全結合層（72だったとき：72＝5308416/64（バッチサイズ）/16の平方根）
        self.dropout = nn.Dropout(p=0.5)  # ドロップアウト:(p=ドロップアウト率)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)#ここも変える
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def predict(img):
    # モデルへの入力
    img = img.convert("RGB")
    img = img.resize((img_size, img_size))
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))  # 平均値を0、標準偏差を1に
                                ])
    img = transform(img)
    x = img.reshape(1, 3, img_size, img_size)

    # 訓練済みモデル
    net = Net()
    net.load_state_dict(torch.load(
        "model_cnn.pth", map_location=torch.device("cpu")
        ))

    # 予測
    net.eval()
    y = net(x)

    # 結果を返す
    y_prob = torch.nn.functional.softmax(torch.squeeze(y))  # 確率で表す
    sorted_prob, sorted_indices = torch.sort(y_prob, descending=True)  # 降順にソート
    return [(classes_ja[idx], classes_en[idx], prob.item()) for idx, prob in zip(sorted_indices, sorted_prob)]
