# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import division
from __future__ import print_function
import torch
import utils
import copy
import networks
import numpy as np
import torch.nn.functional as F
import torch.autograd as autograd
import os

ALGORITHMS = ['ERM', 'ICRM', 'ARM_CML', 'TENT', 'Mixup', 'Fish', 'IB_ERM', 'IB_IRM']

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, hparams):
        super(Algorithm, self).__init__()
        self.num_classes = num_classes
        self.hparams = hparams
        self.metrics = hparams['metrics']
        self.loss_func = utils.get_loss_function(hparams['loss'])
        self.device = hparams['device']
        self.n_tasks_per_step = hparams['n_sampled_tasks'] if hparams.get('n_sampled_tasks') else 0
        
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(self.featurizer.n_outputs, num_classes, self.hparams)
        
        if hparams['is_parallel']:
            print('=> Using data parallel')
            self.featurizer = utils.data_parallel(self.featurizer)
            self.classifier = utils.data_parallel(self.classifier)
                
        self.network = torch.nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = utils.extract_optimizer(self.hparams['optimizer_name'], self.network.parameters(), lr = self.hparams['lr'], weight_decay = self.hparams['weight_decay'])
        
        
    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x, y = None, model = None):
        raise NotImplementedError
    
    def evaluate(self, loader, weights = None, metrics = ['accuracy']):
        raise NotImplementedError()
    
from torch.cuda.amp import autocast, GradScaler
class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    def __init__(self, input_shape, num_classes, hparams):
        super(ERM, self).__init__(input_shape, num_classes,
                                  hparams)
        self.scaler = GradScaler()

    
    # def update(self, minibatches, unlabeled=None):
    #     self.network.train()
    #     # self.train() 
    #     if self.n_tasks_per_step != 0: #步训练时只选择一部分任务进行训练
    #         indices = torch.randperm(len(minibatches))[:min(self.n_tasks_per_step, len(minibatches))]
    #         minibatches = [minibatches[i] for i in indices] #根据随机选择的索引，选择批次数据。



    #     all_y, p = [], [] #分别用于存储所有批次的真实标签（y）和模型的预测输出（p）。
    #       # 1) 清零梯度
    #     self.optimizer.zero_grad()

    #     # 2) AMP 前向过程
    #     with autocast():
    #         for _, (x,y) in enumerate(minibatches):
    #             utils.d_print('Train input shape is', x.shape, y.shape)
    #             # print("开始训练了哈")
    #             out = self.predict(x, y) #执行前向传播
    #             utils.d_print('Output shape is', out.shape)
    #             p.append(out); all_y.append(y)
        
    #         p = torch.cat(p, dim=0)
    #         all_y = torch.cat(all_y, dim=0).view(-1).long()        
    #         loss = self.loss_func(p, all_y)
        
    #     # Compute performance metrics
    #     metric_results = utils.compute_metric(self.metrics, p, all_y)#计算性能指标
    #     metric_results = {'train_' + key:val for key, val in metric_results.items()}
    #     metric_results['train_loss'] = loss.item()

    #     # # Model parameters update
    #     # self.optimizer.zero_grad()
    #     # loss.backward()
    #     # self.optimizer.step()
    #       # 4) AMP 反向+优化
    #     # scale the loss, call backward, then unscale & step
    #     self.scaler.scale(loss).backward()
    #     self.scaler.step(self.optimizer)
    #     self.scaler.update()
    #     return metric_results
    def update(self, minibatches, unlabeled=None):
    # 确保模型在训练模式
        self.network.train()

        p_list = []
        y_list = []

        # 清零梯度
        self.optimizer.zero_grad()

        with autocast():
            for _, (x, y) in enumerate(minibatches):
                # 前向预测
                out = self.predict(x, y)  # out.shape = [full, num_classes]
                p_list.append(out)

                # 按照 predict() 里相同的规则裁剪标签
                B = y.size(0)
                cl = self.train_context_len
                full = (B // cl) * cl
                y_list.append(y[:full])

            # 拼接所有小批次
            p = torch.cat(p_list, dim=0)                # [∑full_i, num_classes]
            all_y = torch.cat(y_list, dim=0).long()     # [∑full_i]

            # 计算 loss
            loss = self.loss_func(p, all_y)

        # 反向传播 & 优化器更新
        loss.backward()
        self.optimizer.step()

        # 计算并返回指标
        metric_results = utils.compute_metric(self.metrics, p, all_y)
        metric_results = {f'train_{k}': v for k, v in metric_results.items()}
        metric_results['train_loss'] = loss.item()
        return metric_results

    def predict(self, x, y = None, model = None):
        print("不喜喜")
        return (self.network if model is None else model)(x) 

    def _evaluate(self, model, loader, metrics=['accuracy']):
        metrics = metrics or self.metrics
        model.eval()
        result = {key: 0 for key in metrics}
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                p = self.predict(x, y, model=model)
                batch_results = utils.compute_metric(metrics, p, y)
                for metric in metrics:
                    result[metric] += batch_results[metric] * len(y)
                total += len(y)
        for metric in metrics:  result[metric] /= (total + 1e-9)
        model.train()
        return result

    def evaluate(self, loader, n_test_samples = 100, module = 'train', cache = None):
        self.network.eval()
        result = {}
        metric_results = self._evaluate(self.network, loader, self.hparams['metrics'])           
        self.test_ctxt = range(0, 51, 5) if module == 'test' else [0, 25, 50, 75, 100]
        for num_samples in self.test_ctxt:
            result.update({f'{metric}(e-{n_test_samples - num_samples})': metric_results[metric] for metric in self.hparams['metrics']})
        self.network.train()
        return result
    
    def _get_ckpt_metric(self):
        return f'acc(e-0)'
 

class TENT(ERM):
    """Tent: Fully Test-Time Adaptation by Entropy Minimization"""
    def __init__(self, input_shape, num_classes, hparams):
        # Make sure to load weights of a trained ERM model before fine-tuning it with TENT
        super().__init__(input_shape, num_classes, hparams)
        self.n_steps = hparams.get('n_steps', 10)
        self.episodic = hparams.get('episodic', 1)
        print(f'Using episodic {bool(self.episodic)} training with {self.n_steps} steps')
        self.flag = 0
    
    def init_states(self):
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(self.network, self.optimizer)
        self.flag = 1
                
    def _setup_model(self):
        if not self.flag:
            self.o_network = copy.deepcopy(self.network)
            self.o_network.eval()
        self.network = self._configure_model(self.network)
        params, _ = self._collect_params(self.network)
        self.optimizer = utils.extract_optimizer(self.hparams['optimizer_name'], params, lr = self.hparams['lr'], weight_decay = self.hparams['weight_decay'])
    
    def evaluate(self, loader, module = 'train', cache = None):
        if not self.flag:
            self._setup_model()
            self.init_states()
        
        self.reset()
        self._setup_model()

        self.test_ctxt = list(range(0, 51, 5)) if module == 'test' else [25, 50, 75, 100]

        if 0 in self.test_ctxt:
            metric_results = self._evaluate(self.o_network, loader, self.hparams['metrics'])  
            result = {f'{metric}(e-100)': metric_results[metric] for metric in self.hparams['metrics']}
        else:
            result = {}
         
        assert self.n_steps > 0, "Tent requires >= 1 step(s) to forward and update"
        for n_samples in self.test_ctxt:
            if n_samples == 0:
                continue
            self.reset()
            metric_results = {key: 0 for key in self.hparams['metrics']}
            sub_loader = torch.utils.data.DataLoader(loader.dataset, batch_size=n_samples, shuffle=False)
            total = 0
            for _, (x, y) in enumerate(sub_loader):
                x, y = x.to(self.device), y.to(self.device)
                if self.episodic:   
                    self.reset()
                t_loss = []
                for _ in range(self.n_steps):
                    loss, p = self._forward_and_adapt(x, self.network, self.optimizer)    
                    t_loss.append(loss.item())
                batch_results = utils.compute_metric(self.hparams['metrics'], p, y)
                for metric in self.hparams['metrics']:
                    metric_results[metric] += batch_results[metric] * len(y)
                total += len(y)
            for metric in self.hparams['metrics']:  metric_results[metric] /= (total + 1e-9)  
            result.update({f'{metric}(e-{100 - n_samples})': metric_results[metric] for metric in self.hparams['metrics']})       
        return result
    
    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer(self.network, self.optimizer, self.model_state, self.optimizer_state)

    @torch.enable_grad()                                                                    # ensure grads in possible no grad context for testing
    def _forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        outputs = model(x)
        loss = utils.softmax_entropy(outputs).mean(0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss, outputs
       
    @staticmethod
    def copy_model_and_optimizer(model, optimizer):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_state = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        return model_state, optimizer_state

    @staticmethod
    def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
        """Restore the model and optimizer states from copies."""
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
        
    @staticmethod
    def _configure_model(model):
        """Configure model for use with tent."""
        model.train()                                                                       # train mode, because tent optimizes the model to minimize entropy
        model.requires_grad_(False)                                                         # disable grad, to (re-)enable only what tent updates
        for m in model.modules():                                                           # configure norm for tent updates: enable grad + force batch statisics
            if isinstance(m, torch.nn.BatchNorm2d):
                m.requires_grad_(True)
                m.track_running_stats = False                                               # force use of batch stats in train and eval modes
                m.running_mean = None
                m.running_var = None
        return model 
    
    @staticmethod
    def _collect_params(model):
        """Collect the affine scale + shift parameters from batch norms.
        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params, names = [], []
        for nm, m in model.named_modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:                                           # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names
    
    @staticmethod
    def _check_model(model):
        """Check model for compatability with tent."""
        is_training = model.training
        assert is_training, "tent needs train mode: call model.train()"
        param_grads = [p.requires_grad for p in model.parameters()]
        has_any_params = any(param_grads)
        has_all_params = all(param_grads)
        assert has_any_params, "tent needs params to update: check which require grad"
        assert not has_all_params, "tent should not update all params: check which require grad"
        has_bn = any([isinstance(m, torch.nn.BatchNorm2d) for m in model.modules()])
        assert has_bn, "tent needs normalization for its optimization"  
        
        

class ICRM(ERM):
    """
    In Context Learner (ICRM)
    """
    def __init__(self, input_shape, num_classes, hparams):
        self.context_len = hparams['context_length']
        self.train_context_len = 100
        hparams['is_transformer'] = 1
        super(ICRM, self).__init__(input_shape, num_classes,
                                  hparams)
                
        # 打印 Featurizer 结构
        # print("Featurizer structure:")
        # print(self.featurizer)
         # 加载预训练模型的步骤
        # pretrained_model_path = hparams['pretrained_model_path']  # 你可以在 hparams 中设置预训练模型的路径
        # if os.path.exists(pretrained_model_path):
        #     print(f"Loading pretrained model from {pretrained_model_path}...")
        #     pretrained_state_dict = torch.load(pretrained_model_path, map_location=self.device)
        #     # 打印模型的 state_dict 键值，以查看每个层的名称
        #     # for name, param in pretrained_state_dict.items():
        #     #     print(f"Layer: {name}, Shape: {param.shape}")
            
        #     # 1) 给所有 key 加 module. 前缀（如果之前还没做）
        #     new_sd = {}
        #     for k, v in pretrained_state_dict.items():
        #         new_sd[ k] = v

        #     # 2) 拿出第 1 层 conv 的权重，扩成 3 通道
        #     w1 = new_sd['context_net.0.weight']  # [128,1,5,5]
        #     # 方法 2.1：直接复制三份
        #     w1_3c = w1.repeat(1, 3, 1, 1)               # [128,3,5,5]
        #     # （可选）方法 2.2：复制后除以 3，保持数值幅度一致
        #     # w1_3c = w1.repeat(1, 3, 1, 1) / 3

        #     new_sd['context_net.0.weight'] = w1_3c
        #     # new_sd['context_net.0.weight'] = w1
        #     # 给每个 key 前面加上 module.
        #     new_sd = {k: v for k, v in new_sd.items() if k not in ["projector.weight", "projector.bias"]}
        #     # 假设预训练模型只有 featurizer 部分
        #     self.featurizer.load_state_dict(new_sd, strict=True)  # 加载预训练权重，strict=False 允许部分加载
        #     print("Pretrained model loaded successfully!")
        # else:
        #     print(f"No pretrained model found at {pretrained_model_path}.")
        # 以上是colouredMNIST的feat版本
        
        
        # pretrained_path = hparams['pretrained_model_path']
        # raw_sd = torch.load(pretrained_path, map_location=self.device)
        # # print("raw_sd 共有", len(raw_sd), "个条目：")
        # # for k in raw_sd.keys():
        # #     print("-", k)

        # model_sd = self.featurizer.state_dict()
        # model_keys = set(model_sd.keys())
        # fixed_sd = {}
        # prefix = "module.network.features."

        # for k, v in raw_sd.items():
        #     # 1) 去掉开头的 "enc."
        #     if k.startswith("enc."):
        #         stripped = k[len("enc."):]       # 或者用 k.replace("enc.", "", 1)
        #     else:
        #         stripped = k
        #     # 2) 拼上你的 prefix
        #     new_k = prefix + stripped

        #     # 3) 看看拼好的 key 在不在 model 里，在的话就加载
        #     if new_k in model_keys:
        #         fixed_sd[new_k] = v
        #     # else: 自动跳过

        # missing, unexpected = self.featurizer.load_state_dict(fixed_sd, strict=False)
        # print(f"✅ Actually loaded {len(fixed_sd)} weights")
        # print("❗ Missing in loaded (model wants but you didn't give):", missing)
        # print("❗ Unexpected in state (you gave but model doesn't need):", unexpected)
        
        
                # 冻结 featurizer
        # for param in self.featurizer.parameters():
        #     param.requires_grad = False
        # print("✅ Featurizer parameters have been frozen.")

        # self.optimizer = torch.optim.Adam(
        #     filter(lambda p: p.requires_grad, self.parameters()),
        #     lr=self.hparams['lr']
        # )



            
    def predict(self, x, y, return_context = False, past_key_values = None): 
        # print("嘻嘻")
        if x.ndim == 4:                                                             # Splits a batch into multiple sequences with length as the context length                                 
            # bs, c, h, w = x.size()                          
            # bs, ctxt = bs // self.train_context_len, self.train_context_len#将批次大小 bs 分割成多个上下文块（ctxt），self.context_len 是每个上下文块的长度。最终 bs 被重新计算为按上下文长度划分的批次数量。
            # y = y.reshape(bs, ctxt)#将标签 y 重塑为适应新的批次和上下文长度
            B, C, H, W = x.size()
            # 2) 确定训练时的上下文长度
            ctxt = self.train_context_len   # 比如 75

            # —— 在这里插入裁剪 full 的逻辑 —— 
            full = (B // ctxt) * ctxt         # e.g. (100//75)*75 = 75
            x_cut = x[:full]              # 保留前 75 张图
            y_cut = y[:full]              # 保留前 75 个标签

            bs = full // ctxt               # 75//75 = 1

            # 3) 切分成 [bs, cl, C, H, W]
            x = x_cut.view(bs, ctxt, C, H, W)
            y = y_cut.view(bs, ctxt)

            # 4) 合并回 featurizer 要的 4D 输入
            x = x.view(bs * ctxt, C, H, W)
        elif x.ndim == 5:   
            bs, ctxt, c, h, w = x.size()
            x = x.contiguous().view(bs * ctxt, c, h, w)
        else:
            raise NotImplementedError
        
        x = self.featurizer(x)
        # print("对对对到这里啦！！！")                                               
        x = x.reshape(bs, ctxt, -1)    #将特征数据 x 重塑为新的形状。bs 是批次大小，ctxt 是上下文长度                                      
        outputs = self.classifier((x, y, None, past_key_values))  #记录了前面已经处理的数据对当前预测的影响
        p, past = outputs[0], outputs[1] #p的形状是 [batch_size, context_len, n_outputs]，表示每个输入的预测结果 。past包含 GPT-2 模型生成的上下文信息。它通常用于保存每次前向传播计算的中间值，以便在后续的生成任务中加速推理。
        if return_context:  return p, past
        else:   return p.view(-1, p.size(-1))
           
    
    def repeat_past_key_values(self, past_key_values, repeats):                     # process key value cache for computing fast inference
        repeated_past_key_values = []
        for layer_past in past_key_values:
            repeated_layer_past = []
            for tensor in layer_past:
                if tensor is not None:
                    repeated_tensor = tensor.repeat_interleave(repeats=repeats, dim=0)
                else:
                    repeated_tensor = None
                repeated_layer_past.append(repeated_tensor)
            repeated_past_key_values.append(tuple(repeated_layer_past))
        return tuple(repeated_past_key_values)



    # def repeat_past_key_values(self, past_key_values, repeats):
    #     if past_key_values is None:
    #         return None
            
    #     # 检查是否是MosaicTransformer的缓存格式（字典）
    #     if isinstance(past_key_values, dict):
    #         # 处理MosaicTransformer的缓存格式
    #         repeated_past = {}
    #         for key, value in past_key_values.items():
    #             if isinstance(value, torch.Tensor):
    #                 repeated_past[key] = value.repeat_interleave(repeats=repeats, dim=0)
    #             else:
    #                 repeated_past[key] = value
    #         return repeated_past
        
    #     # 原来的GPT2格式处理逻辑
    #     repeated_past_key_values = []
    #     for layer_past in past_key_values:
    #         repeated_layer_past = []
    #         for tensor in layer_past:
    #             if tensor is not None:
    #                 repeated_tensor = tensor.repeat_interleave(repeats=repeats, dim=0)
    #             else:
    #                 repeated_tensor = None
    #             repeated_layer_past.append(repeated_tensor)
    #         repeated_past_key_values.append(tuple(repeated_layer_past))
    #     return tuple(repeated_past_key_values)
    
    
    # def repeat_past_key_values(self, past_key_values, repeats):
    #     if past_key_values is None or repeats == 1:
    #         return past_key_values

    #     # Mosaic: dict
    #     if isinstance(past_key_values, dict):
    #         rep = {}
    #         for k, v in past_key_values.items():
    #             if torch.is_tensor(v):
    #                 rep[k] = v.repeat_interleave(repeats, dim=0)
    #             else:
    #                 rep[k] = v  # 标量/长度等信息原样返回
    #         return rep

    #     # GPT-2: tuple of tuples
    #     rep_layers = []
    #     for layer_past in past_key_values:
    #         rep_layer = []
    #         for t in layer_past:
    #             rep_layer.append(t.repeat_interleave(repeats, dim=0) if t is not None else None)
    #         rep_layers.append(tuple(rep_layer))
    #     return tuple(rep_layers)
    
    
    # def repeat_past_key_values(self,past_key_values, repeats: int):
    #     """
    #     Repeat cached KV along batch dim (dim=0).
    #     支持：
    #     - None
    #     - list/tuple of (k,v) pairs: [(k,v), ...]  (Memory Mosaic / GPT2 均可)
    #     - tuple of tuples (GPT-2 原版)
    #     - dict 里存 tensor 的情况（旧的 cached_embeds 实现）
    #     """
    #     if past_key_values is None or repeats == 1:
    #         return past_key_values

    #     # dict 情况
    #     if isinstance(past_key_values, dict):
    #         rep = {}
    #         for k, v in past_key_values.items():
    #             if torch.is_tensor(v):
    #                 rep[k] = v.repeat_interleave(repeats, dim=0)
    #             else:
    #                 rep[k] = v
    #         return rep

    #     # list / tuple 情况
    #     is_list = isinstance(past_key_values, list)
    #     out_layers = []
    #     for layer_past in past_key_values:
    #         if layer_past is None:
    #             out_layers.append(None)
    #             continue
    #         rep_tensors = []
    #         for t in layer_past:
    #             if t is None:
    #                 rep_tensors.append(None)
    #             else:
    #                 rep_tensors.append(t.repeat_interleave(repeats, dim=0))
    #         out_layers.append(tuple(rep_tensors))

    #     return out_layers if is_list else tuple(out_layers)

    
    def _evaluate_robust(self, model, loader, metrics = ['accuracy'], test_cache = None):  #进行模型的 鲁棒性评估 ,metrics：一个列表，指定评估时使用的指标,test_cache：缓存的测试数据
        test_cache_x, test_cache_y = test_cache #从 test_cache 中提取测试数据和标签。
        assert test_cache_x is not None
        assert test_cache_y is not None
        self.network.eval()#将模型切换到 评估模式
        # self.eval()
        model.eval()
        result = {}
        for context_val in self.test_ctxt:  #遍历 self.test_ctxt 中的每个上下文值。self.test_ctxt 控制评估时使用的上下文长度，可以是不同的上下文长度（例如：0、25、50、75、100）。
            with torch.no_grad():   #在评估过程中禁用梯度计算
                if context_val == 0:    initial_past = None    
                else:
                    _, initial_past = self.predict(test_cache_x[:, :context_val], test_cache_y[:, :context_val], return_context = True)
                    initial_past = self.repeat_past_key_values(initial_past, loader._bs)
                #1. xunlian 2. 测试怎么用的
                all_p, all_y = [],[]
                for _, (x, y) in enumerate(loader):
                    # print(f"[DEBUG] x.ndim = {x.ndim}, x.shape = {tuple(x.shape)}")  
                    # # 如果是 5 维，那上下文长度就是 x.shape[1]  
                    # if x.ndim == 5:
                    #     print(f"[DEBUG] context length = {x.shape[1]}")
                    x, y = x.to(self.device), y.to(self.device)   
                    p, _ = self.predict(x, y, return_context = True, past_key_values = initial_past)   
                    if y.ndim == 1: y_reshape = y.view(y.shape[0]//self.context_len, self.context_len)
                    else:   y_reshape = y
                    all_p.append(p);all_y.append(y_reshape)                    
                
                all_p, all_y = torch.cat(all_p, dim=0), torch.cat(all_y, dim=0)
                # print(">> all_p:", all_p.shape, "→ all_p[:, -1, :]:", all_p[:, -1, :].shape)
                # print(">> all_y:", all_y.shape, "→ all_y[:, -1]:", all_y[:, -1].shape)
                ys_pred = all_p.squeeze()             # [20900, 2]
                ys      = all_y.squeeze()             # [20900]
                metric_results = utils.compute_metric(metrics, ys_pred, ys)

                # metric_results = utils.compute_metric(metrics, all_p[:, -1,:], all_y[:, -1])  
                result.update({f'{metric}(e-{self.context_len - context_val})': metric_results[metric] for metric in metrics})  

        self.network.train()
        model.train()
        return result 
    
    def evaluate(self, loader, module = 'train', cache = None):
        self.test_ctxt = list(range(0, 51, 5)) if module == 'test' else [0, 25, 50, 75, 100]   #定义上下文的长度
        result = self._evaluate_robust(self.network, loader, self.hparams['metrics'],  cache)
        return result
 

class ARM_CML(ERM):
    """ Adaptive Risk Minimization (ARM) - (Context Model)"""
    def __init__(self, input_shape, num_classes, hparams):
        original_input_shape = input_shape
        self.n_context_channels = hparams.get('num_features', 1)
        self.ctxt = hparams['support_size']
        self.orig_ctxt = hparams['support_size']
        self.adapt_bn = hparams['adapt_bn']
        input_shape =  (self.n_context_channels + input_shape[0],) + input_shape[1:]             # Since we concatenate the context with input x
        super(ARM_CML, self).__init__(input_shape, num_classes, hparams)
        if hasattr(networks, hparams['context_net']):                                                
            self.context_net = getattr(networks, hparams['context_net'])(original_input_shape[0], hparams)  
            if hparams['is_parallel']:
                self.context_net = utils.data_parallel(self.context_net)
        else:
            raise NotImplementedError()
        
        #  Joint optimizer for ϕ and 𝜃
        params = list(self.network.parameters()) + list(self.context_net.parameters())
        self.optimizer = utils.extract_optimizer(self.hparams['optimizer_name'], params, lr = self.hparams['lr'], weight_decay = self.hparams['weight_decay'])
        self.hparams['mode'] = 'train'
        
        
    def predict(self, x, y = None, model = None):
        bs, _, h, w = x.shape
        re = bs  % self.ctxt
        if self.hparams['mode'] == 'train':
            assert re == 0, 'During training, makre sure batch size is a multiple of support'
            
        if self.hparams['mode'] == 'test' and re != 0:
            x = torch.cat([x, x[:(self.ctxt - re)].clone()], dim=0) 
            y = torch.cat([y, y[:(self.ctxt - re)].clone()], dim=0) 

        eff_bs, supp_size = len(x) // self.ctxt, self.ctxt
        ctxt = self.context_net(x)
        ctxt = ctxt.reshape(eff_bs, supp_size, self.n_context_channels, h, w).mean(dim=1)    
        ctxt = torch.repeat_interleave(ctxt, repeats = supp_size, dim=0)
        x = torch.cat([x, ctxt], dim=1)
        if self.hparams['mode'] == 'test':
            return self.network(x), y
        return self.network(x)
 
    def _evaluate(self, model, loader, metrics=['accuracy']):
        metrics = metrics or self.metrics
        model.eval()
        self.context_net.eval()
        result = {key: 0 for key in metrics}
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                p, y_modif = self.predict(x, y, model=model)
                batch_results = utils.compute_metric(metrics, p, y_modif)
                for metric in metrics:
                    result[metric] += batch_results[metric] * len(y_modif)
                total += len(y_modif)
            for metric in metrics:  result[metric] /= (total + 1e-9)
            model.train()
            return result
   
    def evaluate(self, loader, module = 'train', cache = None):
        self.hparams['mode'] = 'test'
        self.hparams['test_support'] = self.hparams['test_support'] if self.hparams['test_support'] is not None else self.ctxt         
        result = {}
        self.test_ctxt = range(0, 51, 5) if module == 'test' else [0, 25, 50, 75, 100]
        for supp in self.test_ctxt:  
            if supp == 0:
                self.ctxt = supp + 1
            else:
                self.ctxt = supp
            metric_results = self._evaluate(self.network, loader, self.hparams['metrics'])
            result.update({f'{metric}(e-{self.hparams["test_support"] - supp})': metric_results[metric] for metric in self.hparams['metrics']})    
        self.hparams['mode'] = 'train'
        self.context_net.train()
        self.network.train()
        self.ctxt = self.orig_ctxt
        return result 

 

class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """
    def __init__(self, input_shape, num_classes, hparams):
        super(Mixup, self).__init__(input_shape, num_classes,
                                    hparams)

    def update(self, minibatches, unlabeled=None):
        self.network.train()
        objective = 0
        for (xi, yi), (xj, yj) in utils.random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)
            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'train_loss': objective.item()}
    

class IB_ERM(ERM):
    """Information Bottleneck based ERM on feature with conditionning"""

    def __init__(self, input_shape, num_classes, hparams):
        super(IB_ERM, self).__init__(input_shape, num_classes, hparams)
        self.optimizer = utils.extract_optimizer(self.hparams['optimizer_name'], list(self.featurizer.parameters()) + list(self.classifier.parameters()), lr=self.hparams["lr"], weight_decay=self.hparams['weight_decay'])
        self.register_buffer('update_count', torch.tensor([0]))
        
    @property
    def ib_penalty_weight(self):
        return self.hparams['ib_lambda'] if self.update_count >= self.hparams['ib_penalty_anneal_iters'] else 0.0

    def update(self, minibatches, unlabeled=None):
        self.network.train()

        all_x = torch.cat([x for x, y in minibatches])
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        features_list = torch.split(all_features, [x.shape[0] for x, y in minibatches])
        logits_list = torch.split(all_logits, [x.shape[0] for x, y in minibatches])
        
        nll = torch.mean(torch.stack([F.cross_entropy(logits, y) for logits, (x, y) in zip(logits_list, minibatches)]))
        ib_penalty = torch.mean(torch.stack([features.var(dim=0).mean() for features in features_list]))
        loss = nll + self.ib_penalty_weight * ib_penalty
        
        if self.update_count == self.hparams['ib_penalty_anneal_iters']:
            self.optimizer = utils.extract_optimizer(self.hparams['optimizer_name'], list(self.featurizer.parameters()) + list(self.classifier.parameters()), lr=self.hparams["lr"], weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'train_loss': loss.item(), 'nll': nll.item(), 'IB_penalty': ib_penalty.item()}
       
              
class IB_IRM(ERM):
    """Information Bottleneck based IRM on feature with conditionning"""

    def __init__(self, input_shape, num_classes, hparams):
        super(IB_IRM, self).__init__(input_shape, num_classes,
                                  hparams)
        self.optimizer = utils.extract_optimizer(self.hparams['optimizer_name'], list(self.featurizer.parameters()) + list(self.classifier.parameters()), lr=self.hparams["lr"], weight_decay=self.hparams['weight_decay'])
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1., device=device, requires_grad=True)
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        return torch.sum(grad_1 * grad_2)

    @property
    def irm_penalty_weight(self):
        return self.hparams['irm_lambda'] if self.update_count >= self.hparams['irm_penalty_anneal_iters'] else 1.0

    @property
    def ib_penalty_weight(self):
        return self.hparams['ib_lambda'] if self.update_count >= self.hparams['ib_penalty_anneal_iters'] else 0.0

    def update(self, minibatches, unlabeled=None):
        self.network.train()
        all_x = torch.cat([x for x, y in minibatches])
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        features_list = torch.split(all_features, [x.shape[0] for x, y in minibatches])
        logits_list = torch.split(all_logits, [x.shape[0] for x, y in minibatches])

        nll = torch.mean(torch.stack([F.cross_entropy(logits, y) for logits, (x, y) in zip(logits_list, minibatches)]))
        irm_penalty = torch.mean(torch.stack([self._irm_penalty(logits, y) for logits, (x, y) in zip(logits_list, minibatches)]))
        ib_penalty = torch.mean(torch.stack([features.var(dim=0).mean() for features in features_list]))

        loss = nll + self.irm_penalty_weight * irm_penalty + self.ib_penalty_weight * ib_penalty

        if self.update_count == self.hparams['irm_penalty_anneal_iters'] or self.update_count == self.hparams['ib_penalty_anneal_iters']:
            self.optimizer = utils.extract_optimizer(self.hparams['optimizer_name'], list(self.featurizer.parameters()) + list(self.classifier.parameters()), lr=self.hparams["lr"], weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'train_loss': loss.item(), 'nll': nll.item(), 'IRM_penalty': irm_penalty.item(), 'IB_penalty': ib_penalty.item()}
    

class Fish(ERM):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain
    Generalization, Shi et al. 2021.
    """

    def __init__(self, input_shape, num_classes, hparams):
        super(Fish, self).__init__(input_shape, num_classes,
                                   hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.network = networks.WholeFish(input_shape, num_classes, hparams)
        self.optimizer = utils.extract_optimizer(self.hparams['optimizer_name'], self.network.parameters(), lr=self.hparams["lr"],weight_decay=self.hparams['weight_decay'])
        self.optimizer_inner_state = None

    def create_clone(self, device):
        self.network_inner = networks.WholeFish(self.input_shape, self.num_classes, self.hparams, weights=self.network.state_dict()).to(device)
        self.optimizer_inner = utils.extract_optimizer(self.hparams['optimizer_name'],self.network_inner.parameters(),lr=self.hparams["lr"],weight_decay=self.hparams['weight_decay'])
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = utils.ParamDict(meta_weights)
        inner_weights = utils.ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, unlabeled=None):
        self.network.train()
        self.create_clone(minibatches[0][0].device)
        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(meta_weights=self.network.state_dict(),inner_weights=self.network_inner.state_dict(),lr_meta=self.hparams["meta_lr"])
        self.network.reset_weights(meta_weights)

        return {'train_loss': loss.item()}