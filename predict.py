from docopt import docopt
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import common_config as config
from dataset import TextDataset, text_collate_fn,PredictDataset
from model import CRNN
from ctc_decoder import ctc_decode 
from argparse import ArgumentParser

def predict(crnn, dataloader, label2char, decode_method, beam_size):
    crnn.eval()
    pbar = tqdm(total=len(dataloader), desc="Predict")

    all_preds = []
    with torch.no_grad():
        for data in dataloader:
            device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'

            images = data.to(device)
            # cnn + rnn output logits, and find max of logits 
            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            # set logits as ctc_decode input
            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
                               label2char=label2char)
            all_preds += preds

            pbar.update(1)
        pbar.close()

    return all_preds


def show_result(paths, preds):
    print('\n===== result =====')
    for path, pred in zip(paths, preds):
        text = ''.join(pred)
        print(f'{path} > {text}')


def main():

    images = []
    with open('data/test.txt', 'r') as fr:
            for line in fr.readlines():
                path = line.strip().split(' ')[0]
                images.append(path)

    parser = ArgumentParser()
    # dataset setting
    parser.add_argument('--batch_size', type=int, default=256,
                        help='set the batch size (default: 256)')
    parser.add_argument('--beam_size', type=int, default=10,
                        help='set beam size (default: 10)')
    parser.add_argument('--decode_method', type=str, default='beam_search',
                        metavar='greedy, beam_search ,prefix_beam_search' ,help="set decode method (default: beam_search)")
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/crnn_002000_loss9.920046997070312.pt',
                        help='Reload checkpoint ')

    args = parser.parse_args()

    img_height = config['img_height']
    img_width = config['img_width']


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    predict_dataset = PredictDataset(root_dir = config['data_dir'], txt_path='test.txt'
                                      ,img_height=img_height, img_width=img_width)

    predict_loader = DataLoader(
        dataset=predict_dataset,
        batch_size=args.batch_size,
        shuffle=False)

    num_class = len(TextDataset.LABEL2CHAR) + 1
    crnn = CRNN(1, img_height, img_width, num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])

    crnn.load_state_dict(torch.load(args.checkpoint, map_location=device))
    crnn.to(device)

    preds = predict(crnn, predict_loader, TextDataset.LABEL2CHAR,
                    decode_method=args.decode_method,
                    beam_size=args.beam_size)

    show_result(images, preds)


if __name__ == '__main__':
    main()
