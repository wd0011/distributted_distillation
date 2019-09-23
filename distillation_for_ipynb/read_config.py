import configparser
import os
selfname = os.path.basename(os.path.realpath(__file__))
if 'worker' in selfname:
    rank = selfname.split['worker'][1][0]
elif 'master' in selfname:
    rank = 0
else:
    raise ValueError('unexpected filename, please named your file as master.py or worker#.py')

config = configparser.ConfigParser()
config.read('config.ini')
world_size = config['DEFAULTS']['world_size']
lr = config['DEFAULTS']['lr']
momentum = config['DEFAULTS']['momentum']
weight_decay = config['DEFAULTS']['weight_decay']
batch_size = config['DEFAULTS']['batch_size']
backend = config['DEFAULTS']['backend']
aggregation_method = config['DEFAULTS']['aggregation']
epochs = config['DEFAULTS']['epoch']
no_cuda = config['DEFAULTS']['no_gpu']
seed = config['DEFAULTS']['seed']
data_path = config['DEFAULTS']['datapath']
model_path = config['DEFAULTS']['modelpath']
load_model = config['DEFAULTS']['loadmodel']

