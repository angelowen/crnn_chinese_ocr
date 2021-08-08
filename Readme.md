# Chinese ocr based on crnn(繁體中文文字辨識)
The implementation of CRNN (CNN+GRU/LSTM+CTC) for chinese text recognition.
# CRNN

模型採用CRNN（Convolutional Recurrent Neural Network，卷積循環神經網絡）作為backbone，參考論文《An End-to-End Trainable Neural Network for Image-based Sequence Recognition and ItsApplication to Scene Text Recognition》提出的方法，解決基於圖像的場景文字序列識別問題，其特點是，

1. 可以進行end-to-end的訓練
2. 不需要對樣本數據進行字符分割，可識別任意長度的文本序列
3. 模型速度快、性能好，並且模型很小（參數少）

## 模型架構
![](https://i.imgur.com/e5gmovf.png)
CRNN模型主要由以下三部分組成：
1. convolution Layers(CNN)：從輸入圖像中提取出特徵序列
2. Recurrent Layers(Lstm/Gru)：預測從卷積層獲取的特徵序列的標籤分布；
3. Transcription Layers(CTC)：把從循環層獲取的標籤分布通過去重、整合等操作轉換成最終的識別結果。
### CNN + LSTM 架構圖

```
(cnn) input :(batch, channel, height, width)
(cnn) output:(width, batch, feature=channel*height)

(cnn): Sequential(
    (conv0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (relu0): ReLU(inplace=True)
    (pooling0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (relu1): ReLU(inplace=True)
    (pooling1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv2): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (relu2): ReLU(inplace=True)
    (conv3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (relu3): ReLU(inplace=True)
    (pooling2): MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, dilation=1, ceil_mode=False)
    (conv4): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (batchnorm4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu4): ReLU(inplace=True)
    (conv5): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (batchnorm5): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu5): ReLU(inplace=True)
    (pooling3): MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, dilation=1, ceil_mode=False)
    (conv6): Conv2d(512, 512, kernel_size=(2, 2), stride=(1, 1))
    (relu6): ReLU(inplace=True)
  )
  (map_to_seq): Linear(in_features=1536, out_features=64, bias=True)
  (rnn1): LSTM(64, 256, bidirectional=True)
  (rnn2): LSTM(512, 256, bidirectional=True)
  (dense): Linear(in_features=512, out_features=10440, bias=True)
```
### convolution Layers
**預處理->卷積運算->提取序列特徵**

對輸入圖像先做了縮放處理，把所有輸入圖像縮放到相同寬度與高度，結構類似於VGG，由convolution,Maxpooling所組成，而後也多加了Resnet 模型作為backbone可供替換

提取特徵序列中的向量是在特徵圖上從左到右按照順序生成的，用於作為循環層的輸入，每個特徵向量表示了圖像上一定寬度上的特徵，默認的寬度是1，也就是單個像素。由於CRNN已將輸入圖像縮放到同樣高度了，因此只需按照一定的寬度提取特徵即可

### Recurrent Layers

採用bidirectional LSTM構成，考慮LSTM可用於解決時間序列資料的特性，把序列的 width 當作LSTM 的時間 time steps，預測特徵序列中的每一個特徵向量的標籤分布，預測每個label的機率值，

其中，「Map-to-Sequence」自定義網絡層主要是做Recurrent Layers誤差反饋，與特徵序列的轉換，作為卷積層和循環層之間連接的橋樑，從而將誤差從循環層反饋到卷積層。


### Transcription Layers

在CRNN模型中雙向LSTM網絡層的最後連接上一個CTC模型，從而做到了端對端的識別。所謂CTC模型（Connectionist Temporal Classification，聯接時間分類），主要用於解決輸入數據與給定標籤的對齊問題，可用於執行end-to-end的訓練，輸出不定長的序列結果

由於輸入的自然場景的文字圖像，會因字符間隔、圖像變形等問題，導致同個文字有不同的表現形式，但實際上都是同一個詞，

![](https://i.imgur.com/MvLmUFB.png)

而引入CTC就是主要解決這個問題，通過CTC模型訓練後，對結果中去掉間隔字符、去掉重複字符（如果同個字符連續出現，則表示只有1個字符，如果中間有間隔字符，則表示該字符出現多次）

![](https://i.imgur.com/HdJf5Lh.png)

### Beam search algorithm
**參考**
https://hackmd.io/@shaoeChen/H1y-dM6TM?type=view#3-3Beam-search

如果是greedy search只會計算一個單字的機率，而beam search會考慮多單字，透過超參數B(beam width)來控制考慮幾個單字

* Beam search algorithm
    step1. 輸出10000個概率值，並取前三個單字保存

    step2. 針對上面三個單字各自考慮第二個單字，我們關心的是第一個與第二個單字中有最大機率的組合，因為我們設置B=3，最後會有30000個可能，從30000個可能中選出前3個

    step3. 在第二步的時候已經選擇出三個詞對，每一個輸出都會是下一個輸入，一樣的方式各別選擇在10000個詞彙內最高機率的第三個單字，x為前兩個單字。從30000個詞對中選擇出前三個最高機率單字，如果B的設置為1，那就是每次選擇一個，這種模型就是Greedy search。
    
結論: B的設置愈大，選擇愈多，找到的句子就可能越好，但相對的計算成本也高，實務上較常見的設置為10。不同於DFS與BFS的精度搜尋，Beam search不保證能找到最好的。

## CTC 
**參考**
https://www.ycc.idv.tw/crnn-ctc.html
https://distill.pub/2017/ctc/
https://narcissuscyn.github.io/2018/04/14/CTC-loss/

對齊的預處理要花費大量的人力和時間，而且對齊之後，模型預測出的label只是區域性分類的結果，而無法給出整個序列的輸出結果，往往要對預測出的label做一些後處理才可以得到我們最終想要的結果

使用CTC這樣就不需要對資料對齊和一一標註，並且CTC直接輸出序列預測的概率，不需要預先對資料做對齊，只需要一個輸入序列和一個輸出序列即可以訓練
它只關心預測輸出的序列是否和真實的序列是否接近（相同），而不會關心預測輸出序列中每個結果在時間點上是否和輸入的序列正好對齊。


### CTC LOSS (Connectionist Temporal Classification loss)
**參考**
https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html

loss = ctc_loss(input, target, input_lengths, target_lengths)

CTCLoss 對input與target可能對齊的機率進行求和，產生一個損失值，該值相對於每個input node是可微的。 輸入與目標的對齊被設定為“多對一”，這限制了目標序列的長度，使得它輸出長度小於輸入長度

## quick demo
1. `unzip data.zip`
2. `python train.py`
3. `python predict.py --checkpoint XXX.pt`

## Data Preparation
`unzip data.zip`
* data/img -> train & valid imgs file
* data/img_test -> test imgs file

## Training
`python train.py`

you can change hyperparameter in `config.py`
### Special Skills
1. Training Resnet as backbone insted of VGG -> config.py common_config['resnet']= True 
## Testing
`python predict.py --checkpoint XXX.pt`

you can change hyperparameter in `predict.py`

## Todo
* json讀入 data 到dataloader，圖片Resize長寬及預處理方法選擇
* 加入pretrain model 訓練
* lstm+ attention
* Decoded Method:
    * greedy
    * beam_search (beam_size=10)
    * prefix_beam_search (beam_size=10)	
* CTC Decode優化(beam search) 
    * https://hackmd.io/@shaoeChen/H1y-dM6TM?type=view#3-4Refinements-to-beam-search
* optimizer: adam , lr = onecyclelr ,leaky_relu ,dropout
* ensemble
* Preprocessing
    * Remove the noise from the image
    * Remove the complex background from the image
    * Handle the different lightning condition in the image
## Generate synthetic dataset for Chinese OCR.
* https://github.com/wang-tf/Chinese_OCR_synthetic_data
## Reference
[1] https://nanonets.com/blog/deep-learning-ocr/

[2] https://github.com/GitYCC/crnn-pytorch

[3] https://blog.csdn.net/weixin_40546602/article/details/102778029

[4] https://github.com/wavce/cv_papers/blob/master/OCR/WhatIsWrongInSTR.md