import torch.nn as nn
import torch
import torch.nn.functional as F
from config import train_config as config

class CRNN(nn.Module):

    def __init__(self, img_channel, img_height, img_width, num_class,
                 map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False):
        super(CRNN, self).__init__()

        self.cnn, (output_channel, output_height, output_width) = \
            self._cnn_backbone(img_channel, img_height, img_width, leaky_relu)

        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)
        if config['rnn'] == 'lstm':
            self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
            self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)
        else:
            self.rnn1 = nn.GRU(map_to_seq_hidden, rnn_hidden, bidirectional=True)
            self.rnn2 = nn.GRU(2 * rnn_hidden, rnn_hidden, bidirectional=True)
            
        self.dense = nn.Linear(2 * rnn_hidden, num_class)

    def _cnn_backbone(self, img_channel, img_height, img_width, leaky_relu):
        assert img_height % 16 == 0
        assert img_width % 4 == 0

        channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            # shape of input: (batch, input_channel, height, width)
            input_channel = channels[i]
            output_channel = channels[i+1]

            cnn.add_module(
                f'conv{i}',
                nn.Conv2d(input_channel, output_channel, kernel_sizes[i], strides[i], paddings[i])
            )

            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))

            relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            cnn.add_module(f'relu{i}', relu)

        # size of image: (channel, height, width) = (img_channel, img_height, img_width)
        conv_relu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))
        # (64, img_height // 2, img_width // 2)

        conv_relu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))
        # (128, img_height // 4, img_width // 4)

        conv_relu(2)
        conv_relu(3)
        cnn.add_module(
            'pooling2',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (256, img_height // 8, img_width // 4)

        conv_relu(4, batch_norm=True)
        conv_relu(5, batch_norm=True)
        cnn.add_module(
            'pooling3',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (512, img_height // 16, img_width // 4)

        conv_relu(6)  # (512, img_height // 16 - 1, img_width // 4 - 1)

        output_channel, output_height, output_width = \
            channels[-1], img_height // 16 - 1, img_width // 4 - 1
        return cnn, (output_channel, output_height, output_width)

    def forward(self, images):
        # shape of images: (batch, channel, height, width)

        conv = self.cnn(images)

        batch, channel, height, width = conv.size()# conv: [batch,512,3,49]
        conv = conv.view(batch, channel * height, width) # conv: [3, 1536, 49]
        conv = conv.permute(2, 0, 1)  # (width, batch, feature)
        seq = self.map_to_seq(conv) # seq: [49, batch, 64]
        if config['attention']:
            seq = seq.permute(1,0,2)
            if config['rnn'] == 'lstm':
                self.rnn1 = AttRNN(64, 512*seq.shape[1], batch, 200,True).cuda()
            else :
                self.rnn1 = AttRNN(64, 512*seq.shape[1], batch, 200,False).cuda()
                
        recurrent, _ = self.rnn1(seq) # reccur: [49, batch, 512] ,bidirectional will make output size double
        recurrent, _ = self.rnn2(recurrent) # reccur: [49, batch, 512]

        output = self.dense(recurrent)
        return output  # shape: (seq_len, batch, num_class) ,seq_len is equal to img width after convolution

'''
Resnet18
'''
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel,  stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, ResidualBlock, channel, img_height, img_width, num_class,map_to_seq_hidden=64, rnn_hidden=256,):
        super(ResNet18, self).__init__()
        # self.map = map_to_seq_hidden
        self.map_to_seq = nn.Linear(512*img_height//8, map_to_seq_hidden)
        if config['rnn'] == 'lstm':
            self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
            self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)
        else:
            self.rnn1 = nn.GRU(map_to_seq_hidden, rnn_hidden, bidirectional=True)
            self.rnn2 = nn.GRU(2 * rnn_hidden, rnn_hidden, bidirectional=True)
        self.dense = nn.Linear(2 * rnn_hidden, num_class)
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # Res
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(out)
        # Lstm
        batch, channel, height, width = out.size()
        conv = out.view(batch, channel * height, width) 
        conv = conv.permute(2, 0, 1)
        seq = self.map_to_seq(conv)
        if config['attention']:
            seq = seq.permute(1,0,2)
            if config['rnn'] == 'lstm':
                self.rnn1 = AttRNN(64, 512*seq.shape[1], batch, 200,True).cuda()
                # self.rnn2 = AttRNN(512,512*batch,seq.shape[1],200).cuda()
            else :
                self.rnn1 = AttRNN(64, 512*seq.shape[1], batch, 200,False).cuda()         
        recurrent, _ = self.rnn1(seq) 
        recurrent, _ = self.rnn2(recurrent) 
        output = self.dense(recurrent)
        # output = output.permute(1,0,2)

        return output

class AttRNN(nn.Module):
    def __init__(self, input_size, label_size, batch_size, num_layer=1,lstm = True):
        super(AttRNN, self).__init__()
        if lstm:
            self.blstm = torch.nn.LSTM(input_size, input_size, num_layer, bidirectional=True, batch_first=True)
        else:
            self.bgru = torch.nn.GRU(input_size, input_size, num_layer, bidirectional=True, batch_first=True)
        self.lstm = lstm
        self.h0 = torch.randn(2 * num_layer, batch_size, input_size).cuda()
        self.c0 = torch.randn(2 * num_layer, batch_size, input_size).cuda()
        self.softmax = nn.Softmax(dim=2)
        self.tanh = nn.Tanh()
        self.batch_size = batch_size
        self.hidden_size = input_size
        self.loss = nn.BCELoss()
        self.w = torch.randn(input_size).cuda()

        self.embedding_dropout = nn.Dropout(0.3)
        self.lstm_dropout = nn.Dropout(0.3)
        self.attention_dropout = nn.Dropout(0.5)

        self.fc = nn.Sequential(nn.Linear(input_size, label_size))

    def Att_layer(self, H):
        M = self.tanh(H)
        alpha = self.softmax(torch.bmm(M, self.w.repeat(self.batch_size, 1, 1).transpose(1, 2)))
        res = self.tanh(torch.bmm(alpha.transpose(1,2), H))
        return res

    def forward(self, x_input):
        seq_len = x_input.shape[1]
        x_input = self.embedding_dropout(x_input)
        if self.lstm:
            h, _ = self.blstm(x_input, (self.h0, self.c0))
        else:
            h, _ = self.bgru(x_input, self.h0)
        h = h[:,:,self.hidden_size:] + h[:,:,:self.hidden_size]
        h = self.lstm_dropout(h)
        atth = self.Att_layer(h)
        atth = self.attention_dropout(atth)
        out = self.fc(atth)
        out = self.softmax(out)
        return out.view(seq_len,self.batch_size,512),atth

class RNN(nn.Module):
    def __init__(self,num_class,output_channel,output_height,map_to_seq_hidden=64, rnn_hidden=256,lstm = True):
        super(RNN, self).__init__()
        self.lstm = lstm
        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)
        self.dense = nn.Linear(2 * rnn_hidden, num_class)
    def forward(self, conv):
        batch, channel, height, width = conv.size()# conv: [batch,512,3,49]
        conv = conv.view(batch, channel * height, width) # conv: [3, 1536, 49]
        conv = conv.permute(2, 0, 1)  # (width, batch, feature)
        seq = self.map_to_seq(conv) # seq: [49, batch, 64]

        recurrent, _ = self.rnn1(seq) # reccur: [49, batch, 512] ,bidirectional will make output size double
        recurrent, _ = self.rnn2(recurrent) # reccur: [49, batch, 512]

        output = self.dense(recurrent)
        
        return output