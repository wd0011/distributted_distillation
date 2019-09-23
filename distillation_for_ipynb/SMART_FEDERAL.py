import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config as cfg
import sklearn.metrics

aggregation_method_pool = ["naive", "bsz_average", "weight_average", "distillation"]
'''
For now I have completed three naive approach to aggregate the grad calculated by clients.
    'naive': is simply the mean value of all grads. It will be much better when all Servents have same batch size
    'bsz_average': The aggregation of grads is weighted by the batch size. Basically same as 'naive' approach when all Servents have same batch size
    'weiight_average': The aggregation of grads is weighted by the pre-given weight
'''
class Torch_SF:
    def __init__(self, model, backend, rank, world_size, aggregation_method, device):
        '''
        Torch_SF is the basic class。
        '''
        os.environ["MASTER_ADDR"] = cfg.Master_IP_address  # The IP address is the Master's IP address
        os.environ["MASTER_PORT"] = cfg.Master_Port
        if not isinstance(model, nn.Module):
            raise ValueError("The model parameter must be a torch Module")
        if next(model.parameters()).is_cuda:  # The model must be a CUDA model when our backend is 'nccl'
            if backend != "nccl":
                raise TypeError("Our model is CUDA model, CUDA model only expects 'nccl' backend, but got {}".format(backend))
        else:
            if backend == "nccl":
                raise TypeError("Our model is a CPU model, but our backend is 'nccl' and it can not applied on CPU model, please try 'gloo' or 'mpi' or 'tcp' ")
        self.backend = backend
        self.rank = rank
        self.world_size = world_size
        self.NN = model
        self.device = device


        if not (backend == "nccl" or backend == "gloo" or backend == "mpi" or backend == "tcp"):
            raise TypeError("The backend must be one of 'nccl', 'mpi', 'tcp', 'gloo'. but got {}".format(backend))



        if aggregation_method in aggregation_method_pool:
            self.aggregation_method = aggregation_method
        else:
            raise TypeError("Unknown aggregation method {}, For now we only have {}".format(self.aggregation_method, aggregation_method_pool))

        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    def step(self, *args):
        raise NotImplementedError()

    def train(self):
        self.NN.train()

    def eval(self):
        self.NN.eval()


class Master(Torch_SF):
    def __init__(self, model, backend, rank, world_size, learning_rate, device, aggregation_method="naive", given_weight=None):
        '''
        :param model: The model is a torch Model
        :param backend: The communication backend, could be 'gloo', 'mpi' and 'nccl'. 'nccl' is only used for model trained on GPU
        :param rank: The index used to indicate the hierarchy of participants. Each participant must has a unique rank number and Master's rank must be 0
        :param world_size: The total number of participants.
        :param learning_rate: The learning rate used to update our model
        :param aggregation_method: The method that we used to aggregate the grads calculated by Servent
        :param given_weight: Only used for method 'weight_average'
        '''
        super(Master, self).__init__(model, backend, rank, world_size, aggregation_method, device)
        assert rank == 0, "The Master process's rank must be 0 "
        self.NN_buffer = []


        if self.backend == "nccl":
            for i in range(len(list(self.NN.parameters()))):
                self.NN_buffer.append([nn.Parameter(torch.Tensor(list(self.NN.parameters())[i].size())).to(device) for _ in range(self.world_size)])
        else:
            for i in range(len(list(self.NN.parameters()))):
                self.NN_buffer.append([torch.Tensor(list(self.NN.parameters())[i].size()) for _ in range(self.world_size)])
        self.lr = learning_rate



        if self.aggregation_method == "bsz_average":
            if self.backend == "gloo" or self.backend == "mpi" or self.backend == "tcp":
                self.bsz_buffer = [torch.zeros(1) for _ in range(self.world_size)]
            elif self.backend == "nccl":
                self.bsz_buffer = [torch.zeros(1).to(device) for _ in range(self.world_size)]
        elif self.aggregation_method == "weight_average":
            if type(given_weight) == list:
                assert len(given_weight) == world_size-1, "In the aggregation method, The number of numbers in the given_weight must equals to the number of Servents"
                try:
                    t = sum(given_weight)
                    self.given_weight = [i/t for i in given_weight]
                except:
                    raise TypeError("The aggregation method 'weight_weight' must specify a list of number as weight of each Servent")
            else:
                raise TypeError("The aggregation method 'weight_weight' must specify a list of number as weight of each Servent")

    def step(self, model_buffer=None):
        if self.aggregation_method == 'distillation':
            param = []
            for idx in range(len(model_buffer)):
                param.append(list(model_buffer[idx].parameters()))
                for i in range(len(list(model_buffer[idx].parameters()))):
                    dist.broadcast(tensor=param[idx][i].data, src=0)
            
        else:
            for i in range(len(self.NN_buffer)):
                dist.broadcast(tensor=list(self.NN.parameters())[i].data, src=0)

    def update(self, worker_NN_set=None): #*args should be the test data
        for i in range(len(self.NN_buffer)):
            self.master_recv(data=list(self.NN.parameters())[i].data, buffer=self.NN_buffer[i])

        if self.aggregation_method == "naive":
            self.naive_average_grad()
        elif self.aggregation_method == "bsz_average":
            self.batch_size_average_grad()
        elif self.aggregation_method == "weight_average":
            self.weight_aggregation_method()
        elif self.aggregation_method == "distillation":
            self.distillation(worker_NN_set)
        else:
            raise NotImplementedError

    def master_recv(self, data, buffer):
        if self.backend == "gloo" or self.backend == "tcp":
            dist.gather(tensor=data, gather_list=buffer, dst=0)
        elif self.backend == "nccl":
            dist.all_gather(tensor=data, tensor_list=buffer)
        else:
            raise ValueError("Expect the backend 'gloo', 'tcp' or 'nccl', got {}".format(self.backend))

    def weight_aggregation_method(self):
        param = list(self.NN.parameters())
        for i in range(len(self.NN_buffer)):
            grad = torch.zeros_like(self.NN_buffer[i][0])
            for j in range(1, len(self.NN_buffer[i])):
                grad += self.given_weight[j-1] * self.NN_buffer[i][j]
            param[i].data.add_(-self.lr, grad)

    def batch_size_average_grad(self):
        self.master_recv(data=self.bsz_buffer[0], buffer=self.bsz_buffer)
        param = list(self.NN.parameters())
        self.bsz_buffer[0] = sum(self.bsz_buffer[1:])
        for i in range(len(self.NN_buffer)):
            grad = torch.zeros_like(self.NN_buffer[i][0])
            for j in range(1, len(self.NN_buffer[i])):
                factor = self.bsz_buffer[j] / self.bsz_buffer[0]
                grad += factor * self.NN_buffer[i][j]
            param[i].data.add_(-self.lr, grad/self.bsz_buffer[0])

    def naive_average_grad(self):
        param = list(self.NN.parameters())
        for i in range(len(self.NN_buffer)):
            #这里需要改成传过来整个grad好了
            grad = sum(self.NN_buffer[i][1:]) / (len(self.NN_buffer) - 1)
            param[i].data.add_(grad)

    def distillation(self, worker_NN_set):
        param = []
        for i in range(self.world_size-1):
            param.append(list(worker_NN_set[i].parameters()))
            for j in range(len(self.NN_buffer)):
                grad = self.NN_buffer[j][i+1]
                param[i][j].data.add_(grad)



class Servent(Torch_SF):
    def __init__(self, model, backend, rank, world_size, aggregation_method, device):
        super(Servent, self).__init__(model, backend, rank, world_size, aggregation_method, device)
        self.backend = backend
        self.world_size = world_size
        if self.backend == 'nccl':
            self.NN_buffer = []
            for i in range(len(list(self.NN.parameters()))):
                self.NN_buffer.append([nn.Parameter(torch.Tensor(list(self.NN.parameters())[i].size())).to(device) for _ in range(self.world_size)])

        if self.aggregation_method == "bsz_average":  # In the Batch Size average method, we must send the batch size we used to train this model.
            if self.backend == "gloo" or self.backend == "tcp":
                self.bsz_buffer = [torch.zeros(1) for _ in range(self.world_size)]
            elif self.backend == "nccl":
                self.bsz_buffer = [torch.zeros(1).to(device) for _ in range(self.world_size)]

    def step(self, rank, model_buffer=None):
        if self.aggregation_method == 'distillation':
            param = []
            for idx in range(len(model_buffer)):
                param.append(list(model_buffer[idx].parameters()))
                for i in range(len(list(model_buffer[idx].parameters()))):
                    dist.broadcast(tensor=param[idx][i].data, src=0)
            selfdict = model_buffer[rank].state_dict()
            self.NN.load_state_dict(selfdict)
        else:
            for i in range(len(list(self.NN.parameters()))):
                dist.broadcast(tensor=list(self.NN.parameters())[i].data, src=0)


    def batch_size_update(self, batch_size):
        assert self.aggregation_method == "bsz_average", "Only aggregation method 'bsz_average' requires function batch_size_update "
        if self.backend == "gloo" or self.backend == 'tcp':
            bsz_tensor = torch.zeros(1)
            bsz_tensor[0] = float(batch_size)
            self.servent_send(data=bsz_tensor, buffer=None)
        elif self.backend == "nccl":
            bsz_tensor = torch.zeros(1).to(device)
            bsz_tensor[0] = float(batch_size)
            self.servent_send(data=bsz_tensor, buffer=self.bsz_buffer)

    def update(self, backup_model):
        for i in range(len(list(self.NN.parameters()))):
            if self.backend == 'nccl':
                self.servent_send(data=list(self.NN.parameters())[i].data-list(backup_model.parameters())[i].data,
                                  buffer=self.NN_buffer[i])
            else:
                self.servent_send(data=list(self.NN.parameters())[i].data-list(backup_model.parameters())[i].data,
                                  buffer=None)


        #就是做一个buffer，然后将更新好的网络和没更新过的网络的参数相减，作为gradient，然后传到master去

    def servent_send(self, data, buffer):  # The function used for communication between Master and Servent
        if self.backend == "gloo" or self.backend == 'tcp':
            if torch.__version__ > '0.4.1':
                dist.gather(tensor=data, gather_list=[], dst=0)
            else:
                dist.gather(tensor=data, dst=0)
        elif self.backend == "nccl":
            dist.all_gather(tensor=data, tensor_list=buffer)
        else:
            raise ValueError("Expect the backend 'gloo', 'tcp' or 'nccl', got {}".format(self.backend))
