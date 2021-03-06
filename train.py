import os
import cv2
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CTCLoss

from dataset import TextDataset, text_collate_fn
from model import CRNN,ResNet18,ResidualBlock
from evaluate import evaluate
from config import train_config as config
from utils import TPSSpatialTransformerNetwork

def train_batch(crnn, data, optimizer, criterion, device):
    crnn.train()
    images, targets, target_lengths = [d.to(device) for d in data]

    if config['tps-stn']:
        batch,channel,h,w =images.shape
        images = images.permute(0,1,3,2)
        tps = TPSSpatialTransformerNetwork(6, (w, h), (w, h), channel).cuda()
        images = tps(images)
        
    logits = crnn(images)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)


    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    target_lengths = torch.flatten(target_lengths)
    # calculate loss between output and target
    # 藉由輸入每一個橫向pixel的機率值，計算可能變成target的valid alignment 機率值，並將其相加作為loss
    loss = criterion(log_probs, targets, input_lengths, target_lengths)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    epochs = config['epochs']
    train_batch_size = config['train_batch_size']
    eval_batch_size = config['eval_batch_size']
    lr = config['lr']
    show_interval = config['show_interval']
    save_interval = config['save_interval']
    cpu_workers = config['cpu_workers']
    reload_checkpoint = config['reload_checkpoint']

    img_width = config['img_width']
    img_height = config['img_height']
    data_dir = config['data_dir']

    best = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    train_dataset = TextDataset(root_dir=data_dir,txt_path='train.txt', 
                                    img_height=img_height, img_width=img_width)
    valid_dataset = TextDataset(root_dir=data_dir,txt_path='valid.txt',
                                    img_height=img_height, img_width=img_width)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=text_collate_fn)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=eval_batch_size,
        shuffle=True,
        collate_fn=text_collate_fn)

    num_class = len(TextDataset.LABEL2CHAR) + 1 # char to label start from 1
    if config['resnet']:
        crnn = ResNet18(ResidualBlock,1, img_height, img_width,num_class)
    else:        
        crnn = CRNN(1, img_height, img_width, num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])
    if reload_checkpoint:
        crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)

    optimizer = optim.RMSprop(crnn.parameters(), lr=lr)
    criterion = CTCLoss(reduction='sum')
    criterion.to(device)

    # assert save_interval % valid_interval == 0
    # i = 1
    for epoch in range(1, epochs + 1):
        print(f'epoch: {epoch}')
        tot_train_loss = 0.
        tot_train_count = 0
        i = 1
        for train_data in train_loader:
            loss = train_batch(crnn, train_data, optimizer, criterion, device)
            train_size = train_data[0].size(0)

            tot_train_loss += loss
            tot_train_count += train_size
            if i % show_interval == 0:
                print('train_batch_loss[', i, ']: ', loss / train_size)

            i += 1

        evaluation = evaluate(crnn, valid_loader, criterion,
                                      decode_method=config['decode_method'],
                                      beam_size=config['beam_size'])
        print(f'train_loss: {tot_train_loss / tot_train_count} \n','valid_loss={loss}, acc={acc}\n'.format(**evaluation))

        if evaluation['acc'] > best:
            best = evaluation['acc']
            save_model_path = os.path.join(config['checkpoints_dir'],
                                            f'best_model.pt')
            torch.save(crnn.state_dict(), save_model_path)
            print("Save best model~")

        if epoch % save_interval == 0:
            prefix = 'crnn'
            loss = evaluation['loss']
            save_model_path = os.path.join(config['checkpoints_dir'],
                                            f'{prefix}_{epoch}_loss{loss}.pt')
            torch.save(crnn.state_dict(), save_model_path)
            print('save model at ', save_model_path)    



if __name__ == '__main__':
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    main()
