
common_config = {
    'data_dir': './data_color', # ./data_color
    'img_width': 200,
    'img_height': 64,
    'map_to_seq_hidden': 64,
    'rnn_hidden': 256,
    'leaky_relu': True,
    'tps-stn' : False,
    'resnet' : True, 
    'rnn' : 'gru', #'gru','lstm'   all using bidirectional
    'attention' : False,
}


train_config = {
    'epochs': 4000,
    'train_batch_size': 64,
    'eval_batch_size': 512,
    'lr': 0.0005,
    'show_interval': 10,
    'save_interval': 50,
    'cpu_workers': 4,
    'reload_checkpoint': None,
    'decode_method': 'greedy',
    'beam_size': 10,
    'checkpoints_dir': 'checkpoints/'
}
train_config.update(common_config)

