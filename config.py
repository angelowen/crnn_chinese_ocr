
common_config = {
    'data_dir': './data',
    'img_width': 200,
    'img_height': 64,
    'map_to_seq_hidden': 64,
    'rnn_hidden': 256,
    'leaky_relu': False,
    'tps-stn' : True,
    'resnet' : False, 
    'rnn' : 'gru', #'gru','lstm'   all using bidirectional
    'attention' : False
}


train_config = {
    'epochs': 4000,
    'train_batch_size': 32,
    'eval_batch_size': 512,
    'lr': 0.0005,
    'show_interval': 10,
    'valid_interval': 500,
    'save_interval': 2000,
    'cpu_workers': 4,
    'reload_checkpoint': None,
    'valid_max_iter': 100,
    'decode_method': 'greedy',
    'beam_size': 10,
    'checkpoints_dir': 'checkpoints/'
}
train_config.update(common_config)

