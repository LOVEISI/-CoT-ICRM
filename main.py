# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.import os

import os
import sys
import time
import argparse  #解析命令行参数
import json      
import random
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
import utils
import dataset as dataset_file
import hparams_registry     #自定义模块，用于注册和管理模型的超参数（如学习率、批量大小等
import algorithms
import warnings
import string
import datetime
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('=> Home device: {}'.format(device))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default ='~/data/')
    parser.add_argument('--dataset', type=str, default="FEMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=42,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding shuffle dataset and '
        'random_hparams).')         #要用于为数据集的随机打乱和随机超参数生成设置随机种子
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=2001,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=30000,
        help='Checkpoint every N steps. Default is dataset-dependent.'
        )           #控制保存模型检查点的频率
    parser.add_argument('--print_freq', type=int, default=50,
        help='Printing after how many steps. Default is dataset-dependent.'
        )        #控制训练过程中打印信息的频率
    parser.add_argument('--eval_freq', type=int, default=50,
        help='Testing after how many steps. Default is dataset-dependent.'
        )      #每经过 N 步进行一次模型评估
    parser.add_argument('--test_eval_freq', type=int, default=50,
        help='Testing after how many steps. Default is dataset-dependent.'
        )        
    parser.add_argument('--test-args', default={},
        help='arguments for testing'
        )#为测试过程提供额外的参数
    parser.add_argument('--custom-name', default='', type=str, help='custom log names to add') #为日志文件设置一个自定义名称。通常用于为不同的实验指定独特的名称
    parser.add_argument('--colwidth', type=int, default=15) #指定列宽，可能用于打印日志或结果时控制输出格式
    parser.add_argument('--test_envs', type=int, nargs='+', default=[None])
    #指定一个或多个测试环境（通过整数值表示）
    parser.add_argument('--output_dir', type=str, default="~/ICRM/results/") #指定结果输出目录。结果和模型会保存到该目录
    parser.add_argument('--skip_model_save', action='store_true')  #如果设置此参数，程序将跳过模型保存步骤
    parser.add_argument('--wandb', action='store_true', help = 'log into wandb')#启用 wandb（Weights & Biases）日志记录。用于实验跟踪和可视化
    parser.add_argument('--run-name', type=str, default='', help='Choose name of wandb run')#为 wandb 运行设置名称
    parser.add_argument('--project', default='', type=str, help='wandb project dataset_name')#指定 wandb 项目名称
    parser.add_argument('--user', default='', type=str, help='wandb username')
    parser.add_argument('--resume', action='store_true', help='resume wandb previous run')
    parser.add_argument('--run_id', default='', type=str, help='Run ID for Wandb model to resume')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')#每次进行模型检查点保存时都保存模型。启用后，模型将在每个检查点时保存。
    parser.add_argument('--is_parallel', action='store_false') #指示是否启用并行计算
    parser.add_argument('--show_env_results', action='store_true') #显示环境结果。在训练或测试时，可以启用此参数以显示当前环境的结果
    parser.add_argument('--sweep', default=0, type=int, help='sweep mode or not')#用于控制是否启用实验的超参数搜索
    parser.add_argument('--mode', default='train', type=str, help='Training or inference', choices=['train', 'test']) #指定程序的运行模式，支持 train（训练）或 test（推理/测试）两种模式。


    args = parser.parse_args()  #解析命令行传入的参数，将它们存储在 args 对象中
    start_step = 0
    
    if not args.sweep:
        if(args.run_name == ''):     #确保每次实验的运行名称都是唯一的
            args.run_name = f'{args.dataset}-{args.algorithm}-'
            args.run_name += ''.join(random.choice(string.ascii_letters) for i in range(10)) + '-' + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        args.output_dir = os.path.join(args.output_dir, args.dataset)
        args.output_dir = os.path.join(args.output_dir, args.run_name, f'seed-{args.trial_seed}')    #继续拼接输出目录路径
    
    os.makedirs(args.output_dir, exist_ok=True) #使用了 exist_ok=True 参数，表示如果目标目录已经存在，则不会引发错误
    if args.mode == 'train':
        sys.stdout = utils.Tee(os.path.join(args.output_dir, 'out.txt'))
        sys.stderr = utils.Tee(os.path.join(args.output_dir, 'err.txt'))#将训练过程中的标准输出和标准错误输出都保存到文件中
    
    print("=> Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))
    
    ## Path to save run artifacts
    os.environ['WANDB_DIR'] = args.output_dir  #指定 wandb 用来保存实验日志和文件的目录
    if args.wandb:
        logger=utils.Logger(args)      #保存日志
    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset) #根据算法（args.algorithm）和数据集（args.dataset）来选择一组默认的超参数。
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset, utils.seed_hash(args.hparams_seed, args.trial_seed)) #根据指定的超参数种子（args.hparams_seed）和实验种子（args.trial_seed），使用 random_hparams 函数生成一组随机的超参数。
    # hparams['gpt2_bin_path'] = '/mnt/data02/gll_yong/ICRM/pytorch_model_gpt2.bin'

# 这里加入 pretrained_model_path 到 hparams
    # hparams['pretrained_model_path'] = '/mnt/data02/gll_yong/ICRM/mmmlp_best.pth'  # 设置你的预训练模型路径
    # hparams['pretrained_model_path'] = '/root/autodl-tmp/.autodl/FeAT/ColoredMNIST/mmmlp_best.pth'  # 设置你的预训练模型路径
    # hparams['use_mosaic'] = True

    if args.hparams:
        hparams.update(json.loads(args.hparams))   #如果 args.hparams（命令行参数中的超参数设置）不为空，则使用 json.loads() 将其解析为 Python 字典，并更新现有的 hparams 字典。
    
    hparams['device'] = device
    hparams['output_dir'] = args.output_dir
    hparams['overall_seed'] = args.seed  #设置整体的随机种子（args.seed），以确保实验的可重复性。
    hparams['is_parallel'] = args.is_parallel
    hparams['trial_seed'] = args.trial_seed
    hparams['terminal_command'] = " ".join(sys.argv) #记录当前运行的命令行参数，生成的字符串将包含执行脚本时使用的所有参数。这可以用于实验的复现
    print('=> HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))
        
    utils.set_seed(args.seed, device=="cuda")
    if hasattr(dataset_file, args.dataset):       #根据传入的 args.dataset 来加载对应的数据集
        dataset = getattr(dataset_file, args.dataset)(args.data_dir, args.test_envs, hparams)               
    else:
        raise NotImplementedError 

    train_loaders = [utils.InfiniteDataLoader(
                    dataset=env,
                    weights=None,
                    batch_size=hparams['batch_size'],
                    num_workers=dataset.N_WORKERS) #N_WORKERS 表示数据加载时使用的线程数。
                    for i, env in enumerate(dataset)] # 使用 enumerate(dataset) 遍历 dataset 中的每个环境（env）。dataset 可能是一个包含多个训练环境的数据集对象，每个环境可能有不同的数据分布或不同的训练策略。
    
    val_loaders = [utils.FastDataLoader(
                        dataset=env,
                        batch_size=hparams['test_batch_size'] if hparams.get('test_batch_size') is not None else hparams['batch_size'],
                        num_workers=dataset.N_WORKERS,
                        drop_last=True)    #当数据集的大小不能被批量大小整除时，丢弃最后一个不完整的批次
                        for i, env in enumerate(dataset.validation)]

    if dataset.holdout_test:    #创建测试数据加载器
        test_loaders = [utils.FastDataLoader(
                        dataset=env,
                        batch_size=hparams['test_batch_size'] if hparams.get('test_batch_size') is not None else hparams['batch_size'],
                        num_workers=dataset.N_WORKERS,
                        drop_last=True)
                        for i, env in enumerate(dataset.holdout_test)]
   


    if args.algorithm == 'ICRM':#根据所选择的算法类型（args.algorithm）来处理 验证集 和 测试集 数据缓存
        validation_cache = [(x.to(device), y.to(device)) for x, y in zip(dataset.valid_cache_x,dataset.valid_cache_y) ] #zip 会生成一个元组列表，每个元组包含一个特征和一个标签
        holdout_test_cache = [(x.to(device), y.to(device)) for x, y in zip(dataset.test_cache_x,dataset.test_cache_y) ]
    else:
        validation_cache, holdout_test_cache =  [(None, None) for i in dataset.valid_cache_x], [(None, None) for i in dataset.test_cache_x]#这个操作是为了确保 validation_cache 与 dataset.valid_cache_x 的长度相同
            
    algorithm_class = algorithms.get_algorithm_class(args.algorithm) #获取指定的算法类
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, hparams)#返回一个算法实例
    
    # # 打印 featurizer 的初始参数（加载预训练模型之前）
    # print("Featurizer weights before training:")
    # for name, param in  algorithm.featurizer.named_parameters():
    #     print(f"{name}: {param.data.shape} - {param.data.mean()}")
           
    # Resume run
    if os.path.exists(os.path.join(args.output_dir, 'models', 'checkpoint.pth.tar')): #如果存在，则加载检查点并恢复训练状态
        ckpt = utils.load_checkpoint(os.path.join(args.output_dir, 'models'), epoch = None)   
        algorithm.load_state_dict(ckpt['state_dict']) #恢复模型的权重
        start_step = ckpt['results']['step'] #从检查点中提取恢复的步骤数。ckpt['results']['step'] 表示从检查点文件中获取的训练步骤数。
        algorithm.to(device)
        
        algorithm.optimizer.load_state_dict(ckpt['optimizer']) #恢复优化器的状态，ckpt['optimizer'] 包含了保存的优化器状态（如动量、学习率等）
        hparams['best_va'] = ckpt['model_hparams']['best_va']
        hparams['best_te'] = ckpt['model_hparams']['best_te']
        ckpt_metric = ckpt['model_hparams']['best_va'] #从检查点中加载的最佳验证集性能保存到 ckpt_metric 变量中
        print(f'=> Checkpoint loaded and resuming at step {start_step}!')
    else:
        algorithm.to(device)
        hparams['best_va'], hparams['best_te'] = 0, 0

    train_minibatches_iterator = zip(*train_loaders) #创建一个迭代器 train_minibatches_iterator，它将从多个 train_loaders 中并行提取批次数据。
    n_steps = args.steps or dataset.N_STEPS #设置训练的总步骤数
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ #设置模型保存检查点的频率
    ckpt_metric_name = algorithm._get_ckpt_metric() #这个方法返回的指标将用于决定什么时候保存模型检查点。例如，如果 ckpt_metric_name 为 'best_va'，则表示我们基于验证集性能（best_va）来保存检查点。
    args.test_eval_freq = args.eval_freq if args.test_eval_freq is None else args.test_eval_freq
    print(f'=> Checkpointing based on {ckpt_metric_name}') #训练过程中检查点保存的依据是什么
    
    for step in range(start_step, n_steps):
        torch.cuda.empty_cache()
        step_start_time = time.time()
        steps_per_epoch = min([len(env)/hparams['batch_size'] for env in dataset if env not in args.test_envs]) #即训练过程中需要多少个批次来处理整个训练数据集
        info = {'step': step, 'step_time': time.time() - step_start_time}
        
        #  # 打印 featurizer 权重，查看是否有变化
        # if step % 50 == 0:  # 每 50 步打印一次
        #     print(f"Step {step} - Featurizer weights:")
        #     for name, param in algorithm.featurizer.named_parameters():  # 使用 algorithm.featurizer
        #         print(f"{name}: {param.data.shape} - {param.data.mean()}")
        
        minibatches_device = [(x.to(device), y.to(device)) for x,y in next(train_minibatches_iterator)] # 训练数据加载器 train_minibatches_iterator 中获取下一批次数据，并将数据移动到指定设备，每个批次是 (x, y) 对
        step_metrics = algorithm.update(minibatches_device)  #update 方法通常用于执行模型的前向传播、计算损失和更新参数。
        info.update(step_metrics)#info 字典会包含当前步骤的时间以及相关的训练度量


        ## Model evaluation (validation)
        if step % args.eval_freq == 0:
            consolidated_val_results = [] #用于保存各个验证集环境的评估结果
            for index, loader in enumerate(val_loaders):#遍历 val_loaders 中的每个数据加载器，val_loaders 包含多个loader，index 是该加载器的索引,每个loader对应一个client的数据。
                val_metric_results = algorithm.evaluate(loader, cache = validation_cache[index])#val_metric_results 是评估结果，通常是一个字典，包含了各种性能指标（如准确率、损失等）                
                consolidated_val_results.append({f'va_{metric_name}': val for metric_name, val in val_metric_results.items()})
            consolidated_val_results = utils.compute_additional_metrics(hparams.get('additonal_metrics', ['acc']), consolidated_val_results)
            info.update(consolidated_val_results)
            ckpt_metric = consolidated_val_results[f'avg_va_{ckpt_metric_name}']
        
        ## Model evaluation (testing)
        if dataset.holdout_test and step % args.test_eval_freq == 0:
            consolidated_te_results = []
            env_te_results = {}
            for te_index, te_loader in enumerate(test_loaders):
                te_metric_results = algorithm.evaluate(te_loader, cache = holdout_test_cache[te_index])  
                consolidated_te_results.append({f'te_{metric_name}': val for metric_name, val in te_metric_results.items()})  
                if args.show_env_results:
                    env_te_results.update({f'te{te_index}_{metric_name}': val for metric_name, val in te_metric_results.items()})
            consolidated_te_results = utils.compute_additional_metrics(hparams.get('additonal_metrics', ['acc']), consolidated_te_results)
            info.update(consolidated_te_results)
            te_ckpt_metric = consolidated_te_results[f'avg_te_{ckpt_metric_name}']

        # Saving checkpoint and logging metrics (Don't save random training checkpoints)
        if args.save_model_every_checkpoint or step % checkpoint_freq == 0 and False:
            utils.save_checkpoint(algorithm, algorithm.optimizer, hparams, args, info, os.path.join(args.output_dir, 'models'), f'checkpoint_step{step}.pth.tar')
        
        if ckpt_metric >= hparams['best_va'] and step % args.test_eval_freq == 0:#如果当前的验证集指标 ckpt_metric 优于之前记录的最佳验证指标 best_va，并且当前步数满足 test_eval_freq 的要求，则保存当前模型为 最佳模型。
            hparams['best_va'] = ckpt_metric
            hparams['best_te'] = te_ckpt_metric
            utils.save_checkpoint(algorithm, algorithm.optimizer, hparams, args, info, os.path.join(args.output_dir, 'models'), filename = None, save_best = True)
        utils.save_checkpoint(algorithm, algorithm.optimizer, hparams, args, info, os.path.join(args.output_dir, 'models'), filename = 'checkpoint.pth.tar', save_best = False)#如果当前的验证集指标 ckpt_metric 优于之前记录的最佳验证指标 best_va，并且当前步数满足 test_eval_freq 的要求，则保存当前模型为 最佳模型。
   
        info['best_va'], info['best_te'] = hparams['best_va'], hparams['best_te']
        
        # Saving logs for sweeps and collecting results
        if step % args.test_eval_freq == 0:
            save_data = info.copy()
            save_data.update({'hparams': hparams, 'args': vars(args)})
            save_data['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)
            save_data['hparams']['device'] = str(save_data['hparams']['device'])
            with open(os.path.join(args.output_dir, 'results.jsonl'), 'a') as f:
                f.write(json.dumps(save_data, sort_keys=True) + "\n")

        # Model training output      
        if step % args.print_freq == 0 or (step == n_steps - 1):
            if step == 0:
                utils.print_row([i for i in info.keys()], colwidth=args.colwidth)
            utils.print_row([info[key] for key in info.keys()], colwidth=args.colwidth)    
        
        if args.wandb:
            logger.log(info)

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
        f.close()
                
        

    
    