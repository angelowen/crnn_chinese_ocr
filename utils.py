 
import torch
import torch.nn as nn
import torch.nn.functional as function
import numpy as np

# input : [batch_size, channel_num, w, h]
# return: rectified image [batch_size, channel_num, h, w]

class LocalizationNetwork(nn.Module):
    """
     空間變換網絡
     1.讀入輸入圖片，並利用其收集網絡提取特徵
     2.使用特徵計算基準點，基準點的參數由參數指定，通道指定輸入圖像的通道數
     3.計算基準點的方法是使用兩個全連接層將網絡輸出的特徵進行降維，從而得出基準點集合
    """

    def __init__(self, fiducial, channel):
        """
        初始化方法
        :param fiducial: 基準點的數量
        :param channel: 輸入圖像通道數
        """
        super(LocalizationNetwork, self).__init__()
        self.fiducial = fiducial 
        self.channel = channel   
        # 提取特徵使用的捲積網絡
        self.ConvNet = nn.Sequential(
            nn.Conv2d(self.channel, 64, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),  # [N, 64, H, W]
            nn.MaxPool2d(2, 2),  # [N, 64, H/2, W/2]
            nn.Conv2d(64, 128, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),  # [N, 128, H/2, W/2]
            nn.MaxPool2d(2, 2),  # [N, 128, H/4, W/4]
            nn.Conv2d(128, 256, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),  # [N, 256, H/4, W/4]
            nn.MaxPool2d(2, 2),  # [N, 256, H/8, W/8]
            nn.Conv2d(256, 512, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),  # [N, 512, H/8, W/8]
            nn.AdaptiveAvgPool2d(1))  # [N, 512, 1, 1]
        # 計算基準點使用的兩個全連接層
        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256, self.fiducial * 2)
        # 將全連接層2的參數初始化為0
        self.localization_fc2.weight.data.fill_(0)

        ctrl_pts_x = np.linspace(-1.0, 1.0, fiducial // 2)
        ctrl_pts_y_top = np.linspace(0.0, -1.0, fiducial // 2)
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, fiducial // 2)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        # 修改全連接層2的偏移量
        self.localization_fc2.bias.data = torch.from_numpy(initial_bias).float().view(-1)

    def forward(self, x):
        """
        :param x: 輸入圖像，規模[batch_size, C, H, W]
        :return: 輸出基準點集合C，用於圖像校正，規模[batch_size, fiducial, 2]
        """
        batch_size = x.size(0)
        # 提取特徵
        features = self.ConvNet(x).view(batch_size, -1)
        # 使用特徵計算基準點集合C
        features = self.localization_fc1(features)
        C = self.localization_fc2(features).view(batch_size, self.fiducial, 2)
        return C


class GridGenerator(nn.Module):
    """
    Grid Generator of RARE, which produces P_prime by multipling T with P.
    """

    def __init__(self, fiducial, output_size):
        """
        初始化方法
        :param fiducial: 基準點與基本基準點的個數
        :param output_size: 校正後圖像的規模
        基本基準點是被校正後的圖片的基準點集合
        """
        super(GridGenerator, self).__init__()
        self.eps = 1e-6
        # 基準點與基本基準點的個數
        self.fiducial = fiducial
        # 校正後圖像的規模
        self.output_size = output_size # 假設為[w, h]
        # 論文公式當中的C'，C'是基本基準點，也就是被校正後的圖片的基準點集合
        self.C_primer = self._build_C_primer(self.fiducial)
        # 論文公式當中的P'，P'是校正後的圖片的像素坐標集合，規模為[h·w, 2]，集合中有n個元素，每個元素對應校正圖片的一個像素的坐標
        self.P_primer = self._build_P_primer(self.output_size)
        # 如果使用多GPU，則需要暫存器緩存register buffer
        self.register_buffer("inv_delta_C_primer",
                             torch.tensor(self._build_inv_delta_C_primer(self.fiducial, self.C_primer)).float())
        self.register_buffer("P_primer_hat",
                             torch.tensor(
                                 self._build_P_primer_hat(self.fiducial, self.C_primer, self.P_primer)).float())
    def _build_C_primer(self, fiducial):
        """
        構建基本基準點集合C'，即被校正後的圖片的基準點，應該是一個矩形的fiducial個點集合

        :param fiducial: 基本基準點的個數，跟基準點個數相同
        該方法生成C'的方法與前面的空間變換網絡相同
        """
        ctrl_pts_x = np.linspace(-1.0, 1.0, fiducial // 2)
        ctrl_pts_y_top = -1 * np.ones(fiducial // 2)
        ctrl_pts_y_bottom = np.ones(fiducial // 2)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C_primer = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C_primer

    def _build_P_primer(self, output_size):
        """
        構建校正圖像像素坐標集合P'，構建的方法為按照像素靠近中心的程度形成等差數列作為像素橫縱坐標值

        :param output_size: 模型輸出的規模
        :return : 校正圖像的像素坐標集合
        """
        w, h = output_size
        # 等差數列output_grid_x
        output_grid_x = (np.arange(-w, w, 2) + 1.0) / w
        # 等差數列output_grid_y
        output_grid_y = (np.arange(-h, h, 2) + 1.0) / h
        """
        使用np.meshgrid將output_grid_x中每個元素與output_grid_y中每個元素組合形成一個坐標
        注意，若output_grid_x的規模為[w], output_grid_y為[h]，則生成的元素矩陣規模為[h, w, 2]
        """
        P_primer = np.stack(np.meshgrid(output_grid_x, output_grid_y), axis=2)
        # 在返回時將P'進行降維，將P'從[h, w, 2]降為[h·w, 2]
        return P_primer.reshape([-1, 2])  # [HW, 2]

    def _build_inv_delta_C_primer(self, fiducial, C_primer):
        """
        計算deltaC'的inverse，該矩陣為常量矩陣，在確定了fiducial與C'之後deltaC'也同時被確定

        :param fiducial: 基準點與基本基準點的個數
        :param C_primer: 基本基準點集合C'
        :return: deltaC'的inverse
        """
        # 計算C'梯度公式中的R，R中的元素rij等於dij的平方再乘dij的平方的自然對數，dij是C'中第i個元素與C'中第j個元素的歐式距離，R矩陣是個對稱矩陣
        R = np.zeros((fiducial, fiducial), dtype=float)
        # 對稱矩陣可以簡化for循環
        for i in range(0, fiducial):
            for j in range(i, fiducial):
                R[i, j] = R[j, i] = np.linalg.norm(C_primer[i] - C_primer[j])
        np.fill_diagonal(R, 1)  # 填充對稱矩陣對角線元素，都為1
        R = (R ** 2) * np.log(R ** 2)  # 或者R = 2 * (R ** 2) * np.log(R)

        # 使用不同矩陣進行拼接，組成deltaC'
        delta_C_primer = np.concatenate([
            np.concatenate([np.ones((fiducial, 1)), C_primer, R], axis=1),       # 規模[fiducial, 1+2+fiducial]，deltaC'計算公式的第一行
            np.concatenate([np.zeros((1, 3)), np.ones((1, fiducial))], axis=1),  # 規模[1, 3+fiducial]，deltaC'計算公式的第二行
            np.concatenate([np.zeros((2, 3)), np.transpose(C_primer)], axis=1)   # 規模[2, 3+fiducial]，deltaC'計算公式的第三行
        ], axis=0)                                                               # 規模[fiducial+3, fiducial+3]

        # 調用np.linalg.inv求deltaC'的逆
        inv_delta_C_primer = np.linalg.inv(delta_C_primer)
        return inv_delta_C_primer

    def _build_P_primer_hat(self, fiducial, C_primer, P_primer):
        """
        求^P'，即論文公式當中由校正後圖片像素坐標經過變換矩陣T後反推得到的原圖像素坐標P集合公式當中的P'hat，P = T^P'

        :param fiducial: 基準點與基本基準點的個數
        :param C_primer: 基本基準點集合C'，規模[fiducial, 2]
        :param P_primer: 校正圖像的像素坐標集合，規模[h·w, 2]
        :return: ^P'，規模[h·w, fiducial+3]
        """
        n = P_primer.shape[0]  # P_primer的規模為[h·w, 2]，即n=h·w
        # PAPER: d_{i,k} is the euclidean distance between p'_i and c'_k
        P_primer_tile = np.tile(np.expand_dims(P_primer, axis=1), (1, fiducial, 1))  # 規模變化 [h·w, 2] -> [h·w, 1, 2] -> [h·w, fiducial, 2]
        C_primer = np.expand_dims(C_primer, axis=0)                                  # 規模變化 [fiducial, 2] -> [1, fiducial, 2]
        # 此處相減是對於P_primer_tile的每一行都減去C_primer，因為這兩個矩陣規模不一樣
        dist = P_primer_tile - C_primer
        # 計算求^P'公式中的dik，dik為P'中第i個點與C'中第k個點的歐氏距離
        r_norm = np.linalg.norm(dist, ord=2, axis=2, keepdims=False)                 # 規模 [h·w, fiducial]
        # r'ik = d^2ik·lnd^2ik
        r = 2 * np.multiply(np.square(r_norm), np.log(r_norm + self.eps))
        # ^P'i = [1, x'i, y'i, r'i1,......, r'ik]的轉置，k=fiducial
        P_primer_hat = np.concatenate([np.ones((n, 1)), P_primer, r], axis=1)        # 規模 經過垂直拼接[h·w, 1]，[h·w, 2]，[h·w, fiducial]形成[h·w, fiducial+3]
        return P_primer_hat

    def _build_batch_P(self, batch_C):
        """
        求本batch每一張圖片的原圖像素坐標集合P

        :param batch_C: 本batch原圖的基準點集合C
        :return: 本batch的原圖像素坐標集合P，規模[batch_size, h, w, 2]
        """
        
        # 獲取batch_size
        batch_size = batch_C.size(0)
 
        # 將本batch的基準點集合進行擴展，使其規模從[batch_size, fiducial, x] -> [batch_size, fiducial+3, 2]
        batch_C_padding = torch.cat((batch_C, torch.zeros(batch_size, 3, 2).float().cuda()), dim=1)

        # 按照論文求解T的公式求T，規模變化[fiducial+3, fiducial+3] × [batch_size, fiducial+3, 2] -> [batch_size, fiducial+3, 2]
        batch_T = torch.matmul(self.inv_delta_C_primer, batch_C_padding)
        # 按照論文公式求原圖像素坐標的公式求解本batch的原圖像素坐標集合P，P = T^P'
        # [h·w, fiducial+3] × [batch_size, fiducial+3, 2] -> [batch_size, h·w, 2]
        batch_P = torch.matmul(self.P_primer_hat, batch_T)
        # 將P從[batch_size, h·w, 2]轉換到[batch_size, h, w, 2]
        
        return batch_P.reshape([batch_size, self.output_size[1], self.output_size[0], 2])

    def forward(self, batch_C):
        return self._build_batch_P(batch_C)


class TPSSpatialTransformerNetwork(nn.Module):
    """Rectification Network of RARE, namely TPS based STN"""

    def __init__(self, fiducial, input_size, output_size, channel):
        """Based on RARE TPS

        :param fiducial: number of fiducial points
        :param input_size: (w, h) of the input image
        :param output_size: (w, h) of the rectified image
        :param channel: input image channel
        """
        super(TPSSpatialTransformerNetwork, self).__init__()
        self.fiducial = fiducial
        self.input_size = input_size
        self.output_size = output_size
        self.channel = channel
        self.LNet = LocalizationNetwork(self.fiducial, self.channel)
        self.GNet = GridGenerator(self.fiducial, self.output_size)

    def forward(self, x):
        """
        :param x: batch input image [batch_size, c, w, h]
        :return: rectified image [batch_size, c, h, w]
        """

        # 求原圖的基準點集合C
        C = self.LNet(x).cuda()  # [batch_size, fiducial, 2]
        # 求原圖對應校正圖像素的像素坐標集合P
        P = self.GNet(C).cuda() # [batch_size, h, w, 2]

        # 按照P對x進行採樣，對於越界的位置在網格中採用邊界的pixel value進行填充

        rectified = function.grid_sample(x, P, padding_mode='border', align_corners=True)  #規模[batch_size, c, h, w]
        return rectified

