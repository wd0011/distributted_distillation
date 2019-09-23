import configparser

def build_config():
    config = configparser.ConfigParser()

    config['DEFAULTS'] = {'lr':              1e-3,
                         'momentum':        0.01,
                         'weight_decay':    1e-3,
                         'batch_size':      256,
                         'lr_decay':        True, # rule of lr decay should set by hand in utils
                         'world_size':      4,
                         'epoch':           2000,
                         'backend':         'nccl',
                         'aggregation':     'naive',
                         'datapath':        'nobodyknows',
                         'modelpath':       'nobodyknows',
                         'loadmodel':       False,
                         'no_gpu':          False,
                         'seed':            35
                         }

    with open('config.ini', 'w') as configfile:
        config.write(configfile)

if __name__=='__main__':
    build_config()