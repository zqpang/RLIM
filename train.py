from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import numpy
import sys
import collections
import time
import os
from datetime import timedelta
from sklearn.cluster import DBSCAN
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F        

from rlim import datasets
from rlim import models
from rlim.models.memory import Memory
from rlim.trainers import RLIM_USL
from rlim.evaluators import Evaluator, extract_features
from rlim.utils.data import IterLoader
from rlim.utils.data import transforms as T
from rlim.utils.data.sampler import RandomMultipleGallerySampler
from rlim.utils.data.preprocessor import Preprocessor_train, Preprocessor_test
from rlim.utils.logging import Logger
from rlim.utils.serialization import load_checkpoint, save_checkpoint
from rlim.utils.faiss_rerank import compute_jaccard_distance




def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset

def get_train_loader(args, dataset, height, width, batch_size, workers,
                    num_instances, iters, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
	         T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])
    
    train_transformer2 = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.Grayscale(num_output_channels=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
	         T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]),
        ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(Preprocessor_train(train_set, root=dataset.dataset_dir, transform1=train_transformer,transform2 = train_transformer2),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader



def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    train = True
    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))
        train = False

    test_loader = DataLoader(
        Preprocessor_test(testset, train, root=dataset.dataset_dir, transform1=test_transformer,transform2 = test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    return test_loader





def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout, num_classes=0)
    model.cuda()
    model = nn.DataParallel(model)
    return model

def evaluate_mean(evaluator1, dataset, test_loaders):
    maxap = 0
    maxcmc = 0
    mAP_sum = 0
    cmc_sum = 0
    cmc_sum_10 = 0

    for i in range(len(dataset)):
        cmc_scores, mAP = evaluator1.evaluate(test_loaders[i], dataset[i].query, dataset[i].gallery, cmc_flag=False)
        maxap = max(mAP, maxap)
        maxcmc = max(cmc_scores[0], maxcmc)
        mAP_sum += mAP
        cmc_sum += cmc_scores[0]
        cmc_sum_10 += cmc_scores[9]

    mAP = (mAP_sum) / len(test_loaders)
    cmc_now = (cmc_sum) / len(test_loaders)
    cmc_now_10 = cmc_sum_10 / (len(test_loaders))

    return mAP, cmc_now, cmc_now_10

def cluster_finement_new(pseudo_labels, rerank_dist, pseudo_labels_tight):
    rerank_dist_tensor = torch.tensor(rerank_dist)
    N = pseudo_labels.size(0)
    
    label_sim_expand = pseudo_labels.expand(N, N) #N行重复的pseudo_labels
    label_sim_tight_expand = pseudo_labels_tight.expand(N, N) #N行重复的pseudo_labels_tight
    
    label_sim = label_sim_expand.eq(label_sim_expand.t()).float() #每行哪几个索引值为1就是与哪几个索引的样本具有相同的pseudo_labels
    label_sim_tight = label_sim_tight_expand.eq(label_sim_tight_expand.t()).float() #每行哪几个索引值为1就是与哪几个索引的样本具有相同的pseudo_labels_tight
    
    sim_distance = rerank_dist_tensor.clone() * label_sim  #第i行的非0数值表明第i个样本与簇内其他样本的jaccard_distance
    
    sample_num = label_sim.sum(-1)  #第i个索引表示第i个样本所在簇中的样本数量
    dis_num = sample_num.clone()
    dis_num[dis_num > 1] -= 1 #第i个索引表示第i个样本所在簇中除该样本外的样本数量
    
    sample_ave_dis = sim_distance.sum(-1) / dis_num  #第i个索引表示第i个样本与其所在簇内其他样本的距离均值
    
    sample_ave_dis_cal = sample_ave_dis / sample_num
    cluster_I_average = torch.zeros(torch.max(pseudo_labels).item() + 1)
    for sim_dists, label in zip(sample_ave_dis_cal, pseudo_labels):
        cluster_I_average[label.item()] += sim_dists #cluster_I_average的第i个元素值表示第i个簇中两两样本之间的平均距离
    
    sample_num_tight = label_sim_tight.sum(-1)
    sample_ave_dis_cal_tight = sample_ave_dis / sample_num_tight
    
    tight_I_average = torch.zeros(torch.max(pseudo_labels_tight).item() + 1)
    for sim_dists, label in zip(sample_ave_dis_cal_tight, pseudo_labels_tight):
        tight_I_average[label.item()] += sim_dists
    
    return sample_ave_dis, cluster_I_average, tight_I_average



def main():    
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    main_worker(args)
    




def main_worker(args):
    global start_epoch, best_mAP
    best_mAP =0
    best_rank1 = 0
    start_time = time.monotonic()
    cudnn.benchmark = True
    
    sys.stdout = Logger(osp.join(args.logs_dir, args.dataset, 'logs.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    iters = args.iters if (args.iters>0) else None
    print("==> Load unlabeled dataset")
    #args.data_dir = '/root/pxu1/datasets/{}_all'.format(args.dataset)
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, 256, args.workers)
    
    
    
   
    
    encoder = create_model(args)
    #model_mask = create_model(args)
    
    evaluator1 = Evaluator(encoder)
    
    
    memory = Memory(encoder.module.num_features, len(dataset.train), temp=args.temp, bg_knn=args.bg_knn).cuda()
    
    cluster_loader = get_test_loader(dataset, args.height, args.width,
                                    256, args.workers, testset=sorted(dataset.train))

        
    params = []
    print('prepare parameter')
    models = [encoder]
    for model in models:
        for key, value in model.named_parameters():
            if value.requires_grad:
                params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]

    optimizer = torch.optim.Adam(params)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)


    # Trainer
    print('==> Start training')
    trainer = RLIM_USL(encoder, memory)

    # create image-level camera information
    all_img_cams = torch.tensor([c for _, _, c in sorted(dataset.train)])
    temp_all_cams = all_img_cams.numpy()
    all_img_cams = all_img_cams.cuda()
    unique_cameras = torch.unique(all_img_cams)
    
    
    for epoch in range(args.epochs):
        print('\n==> Epoch {}: Create pseudo labels for unlabeled data with self-paced policy'.format(epoch))
        
        features, _ = extract_features(encoder, cluster_loader, print_freq=50)
        features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
        
        #features2, _= extract_features(model_mask, cluster_loader, print_freq=50, judg=2)
        #features2 = torch.cat([features2[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)        
        memory.features = F.normalize(features, dim=1).cuda()
        
        del features
        
        features = memory.features.clone()
        now_time_before_cluster =  time.monotonic()
        
        rerank_dist = compute_jaccard_distance(features.cpu(), k1=args.k1, k2=args.k2)
        
        del features
        
        if (epoch==0):
            params = {
                        'eps': args.eps + args.eps_gap,
                        'eps_tight': args.eps,
                        #'eps_loose': args.eps + args.eps_gap,
                        'min_samples': args.min_samples,
                        'metric': 'precomputed',
                        'n_jobs': -1
                    }
            print('Clustering criterion: eps: {:.3f}, eps_tight: {:.3f}'.format(params['eps'], params['eps_tight']))
            cluster = DBSCAN(eps=params['eps'], min_samples=params['min_samples'], metric=params['metric'], n_jobs=params['n_jobs'])
            cluster_tight = DBSCAN(eps=params['eps_tight'], min_samples=params['min_samples'], metric=params['metric'], n_jobs=params['n_jobs'])
            #cluster_loose = DBSCAN(eps=params['eps_loose'], min_samples=params['min_samples'], metric=params['metric'], n_jobs=params['n_jobs'])
        
        pseudo_labels = cluster.fit_predict(rerank_dist)
        pseudo_labels_tight = cluster_tight.fit_predict(rerank_dist)
        
        
        num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
        num_ids_tight = len(set(pseudo_labels_tight)) - (1 if -1 in pseudo_labels_tight else 0)        
        
        
        def proxy_labels_generate(pseudo_labels, temp_all_cams):
            proxy_labels = -1 * np.ones(pseudo_labels.shape, pseudo_labels.dtype)
            cnt = 0
            for i in range(0, int(pseudo_labels.max() + 1)):
                inds = np.where(pseudo_labels == i)[0]
                local_cams = temp_all_cams[inds]
                for cc in np.unique(local_cams):
                    pc_inds = np.where(local_cams==cc)[0]
                    proxy_labels[inds[pc_inds]] = cnt
                    cnt += 1
            num_proxies = len(set(proxy_labels)) - (1 if -1 in proxy_labels else 0)
            
            return proxy_labels, num_proxies
        
        proxy_labels, num_proxies = proxy_labels_generate(pseudo_labels, temp_all_cams)
        
        proxy_labels_tight, num_proxies_tight = proxy_labels_generate(pseudo_labels_tight, temp_all_cams)
        
        
        def generate_pseudo_labels(cluster_id, num):
            labels = []
            outliers = 0
            for i, ((fname, _, cid), id) in enumerate(zip(sorted(dataset.train), cluster_id)):
                label = id if id != -1 else num + outliers
                labels.append(label)
                if id == -1:
                    outliers += 1
            return torch.Tensor(labels).long(), outliers    
        

        pseudo_labels, _ = generate_pseudo_labels(pseudo_labels, num_ids)
        pseudo_labels_tight, _ = generate_pseudo_labels(pseudo_labels_tight, num_ids_tight)
        
        proxy_labels, proxy_outliers = generate_pseudo_labels(proxy_labels, num_proxies)
        proxy_labels_tight, _ = generate_pseudo_labels(proxy_labels_tight, num_proxies_tight)
        
        N = pseudo_labels.size(0)
        
        pseudo_labels_nums = len(torch.unique(pseudo_labels))
        
        
        index2label = collections.defaultdict(int)
        for label in pseudo_labels:
            index2label[label.item()]+=1
        index2label = np.fromiter(index2label.values(), dtype=float)

        index2pro = collections.defaultdict(int)
        for label in proxy_labels:
            index2pro[label.item()]+=1
        index2pro = np.fromiter(index2pro.values(), dtype=float)
        
        print('==> Statistics for epoch {}: {} clusters, {} sub-clusters, {} un-clustered instances'
                    .format(epoch, (index2label>1).sum(), (index2pro>1).sum()+(index2pro==1).sum()-(index2label==1).sum(), (index2label==1).sum()))
        
        
        
        
        #pseudo_labels_old = pseudo_labels.clone()
        #index2label_old = index2label.copy()
        # =====================================================
        if args.cr: 
            print('cluster refinement')
            pseudo_labeled_dataset = []
            outliers = 0
            sample_ave_dis, cluster_I_average, tight_I_average = cluster_finement_new(proxy_labels, rerank_dist, proxy_labels_tight)
            for i, ((fname, _, cid), pseudo_label, proxy_label, proxy_label_tight) in enumerate(zip(sorted(dataset.train), pseudo_labels, proxy_labels, proxy_labels_tight)):
                
                average = cluster_I_average[proxy_label.item()]
                '''if pseudo_label < num_ids:
                    if  ((args.ratio_cluster * sample_ave_dis[i].item() <= average) or (tight_I_average[proxy_label_tight.item()] <= average)):
                        pseudo_labeled_dataset.append((fname, pseudo_label.item(), proxy_label.item(), cid))
                    else:
                        pseudo_labels[i] = pseudo_labels_nums+outliers
                        proxy_labels[i] = len(cluster_I_average)+outliers
                        outliers+=1'''
                
                if  ((args.ratio_cluster * sample_ave_dis[i].item() <= average) or (tight_I_average[proxy_label_tight.item()] <= average)):
                    pseudo_labeled_dataset.append((fname, pseudo_label.item(), proxy_label.item(), cid))
                else:
                    pseudo_labeled_dataset.append((fname, pseudo_labels_nums+outliers, len(cluster_I_average)+outliers, cid))
                    pseudo_labels[i] = pseudo_labels_nums+outliers
                    proxy_labels[i] = len(cluster_I_average)+outliers
                    outliers+=1
        #  =====================================================
            now_time_after_cluster =  time.monotonic()
            print(
                'the time of cluster refinement is {}'.format(now_time_after_cluster-now_time_before_cluster)
            )
            #print('len(pseudo_labeled_dataset):', len(pseudo_labeled_dataset))


            
        else:
            print('No cluster refinement')
            pseudo_labeled_dataset = []
            '''if pseudo_label < num_ids:
                 for i, ((fname, pid, cid), pseudo_label, proxy_label) in enumerate(zip(sorted(dataset.train), pseudo_labels, proxy_labels)):
                     pseudo_labeled_dataset.append((fname, pseudo_label.item(), proxy_label.item(), cid))'''
                
            for i, ((fname, pid, cid), pseudo_label, proxy_label) in enumerate(zip(sorted(dataset.train), pseudo_labels, proxy_labels)):
                pseudo_labeled_dataset.append((fname, pseudo_label.item(), proxy_label.item(), cid))
        
        
        
        index2label = collections.defaultdict(int)
        for label in pseudo_labels:
            index2label[label.item()]+=1
        index2label = np.fromiter(index2label.values(), dtype=float)
        
        index2pro = collections.defaultdict(int)
        for label in proxy_labels:
            index2pro[label.item()]+=1
        index2pro = np.fromiter(index2pro.values(), dtype=float)        

        print('==> Statistics for epoch {}: {} clusters, {} sub-clusters, {} un-clustered instances'
                    .format(epoch, (index2label>1).sum(), (index2pro>1).sum()+(index2pro==1).sum()-(index2label==1).sum(), (index2label==1).sum()))
        
        
        memory.all_pseudo_label = pseudo_labels.cuda()
        memory.all_proxy_label = proxy_labels.cuda()


        memory.proxy_label_dict = {}  # {pseudo_label1: [proxy3, proxy10],...}
        proxy2cluster = {}
        for c in range(0, int(memory.all_pseudo_label.max() + 1)):
            memory.proxy_label_dict[c] = torch.unique(memory.all_proxy_label[memory.all_pseudo_label == c])
            
            x = torch.unique(memory.all_proxy_label[memory.all_pseudo_label == c])
            permutation = torch.randperm(x.numel())
            proxy2cluster[c] = x.view(-1)[permutation][0]     
        

        memory.proxy_cam_dict = {}  # for computing proxy enhance loss
        for cc in unique_cameras:
            proxy_inds = torch.unique(memory.all_proxy_label[(all_img_cams == cc) & (memory.all_proxy_label>=0)])
            memory.proxy_cam_dict[int(cc)] = proxy_inds
        
        
        
        features = memory.features.clone()
        pseudo_labels_nums = len(torch.unique(pseudo_labels))
        proxy_labels_nums = len(torch.unique(proxy_labels))
        


        ## initialize cluster memory
        cluster_centers = torch.zeros(pseudo_labels_nums, features.size(1))
        for ii in range(pseudo_labels_nums):
            idx = torch.nonzero(pseudo_labels == ii).squeeze(-1)
            cluster_centers[ii] = features[idx].mean(0)
        #cluster_centers = F.normalize(cluster_centers.detach(), dim=1).cuda()
        print('initializing cluster memory feature with shape {}...'.format(cluster_centers.shape))
        memory.cluster_centers = cluster_centers.cuda().detach()



        # initialize proxy memory
        proxy_centers = torch.zeros(proxy_labels_nums, features.size(1))
        for lbl in range(proxy_labels_nums):
            ind = torch.nonzero(proxy_labels == lbl).squeeze(-1)  # note here
            id_feat = features[ind].mean(0)
            proxy_centers[lbl,:] = id_feat
        #proxy_centers = F.normalize(proxy_centers.detach(), dim=1).cuda()
        print('initializing proxy memory feature with shape {}...'.format(proxy_centers.shape))
        memory.proxy_centers = proxy_centers.cuda().detach()
               

        proxy_centers2 = torch.zeros(pseudo_labels_nums, features.size(1))
        for proxy2cluster_i in proxy2cluster:
            proxy_centers2[proxy2cluster_i] = proxy_centers[proxy2cluster[proxy2cluster_i]]
        print('initializing proxy_centers2 with shape {}...'.format(proxy_centers2.shape))
        memory.proxy_centers2 = proxy_centers2.cuda().detach()

        del features




        
        train_loader1 = get_train_loader(args, dataset, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_dataset)
        train_loader1.new_epoch()
        trainer.train(epoch, train_loader1, pseudo_labels_nums, optimizer, print_freq=args.print_freq, train_iters=len(train_loader1))
        
        
        if epoch > 30:
            args.eval_step = 1 
        
        
        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
                
            cmc_socore1,mAP1 = evaluator1.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            mAP, cmc_now, cmc_now_5, cmc_now_10  = mAP1, cmc_socore1[0], cmc_socore1[4], cmc_socore1[9]
            
            print('===============================================')
            print('the model performance')
            print('mAP: {:5.1%}'.format(mAP))
            print('Rank-1: {:5.1%}'.format(cmc_now))
            print('Rank-5: {:5.1%}'.format(cmc_now_5))
            print('Rank-10: {:5.1%}'.format(cmc_now_10))
            print('===============================================')
            
            
            is_best = (mAP>best_mAP)
            #is_bset = (cmc_now > best_rank1)
            best_mAP = max(mAP, best_mAP)
            best_rank1 = max(cmc_now, best_rank1)
            
            
            
            save_checkpoint({
                'state_dict': encoder.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best,fpath=osp.join(args.logs_dir, args.dataset, 'checkpoint.pth.tar'))
            print('\n * Finished epoch {:3d}  model cmc: {:5.1%}  best: {:5.1%}{}\n'.format(epoch, cmc_now, best_rank1, ' *' if is_best else ''))
        lr_scheduler.step()

        
        
    print ('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, args.dataset, 'model_best.pth.tar'))
    encoder.load_state_dict(checkpoint['state_dict'])
    
    cmc_socore1,mAP1 = evaluator1.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
    mAP, cmc_now, cmc_now_5, cmc_now_10  = mAP1, cmc_socore1[0], cmc_socore1[4], cmc_socore1[9]
    
    print('=================RGB===================')
    print('the model performance')
    print('mAP: {:5.1%}'.format(mAP))
    print('Rank-1: {:5.1%}'.format(cmc_now))
    print('Rank-5: {:5.1%}'.format(cmc_now_5))
    print('Rank-10: {:5.1%}'.format(cmc_now_10))
    print('===============================================')
    
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RLIM")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='ltcc',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=0)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.60,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.03,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--min_samples', type=float, default=4,
                        help="min_samples for DBSCAN")    
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--output_weight', type=float, default=1.0,
                        help="loss outputs for weight ")
    parser.add_argument('--ratio_cluster', type=float, default=0.999,
                        help="cluster hypter ratio ")
    parser.add_argument('--cr', action="store_true", default=False,
                        help="use cluster refinement")
    
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--bg_knn', type=int, default=50)    
    parser.add_argument('--loss-size', type=int, default=2)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    parser.add_argument('--sic_weight', type=float, default=1,
                        help="loss outputs for sic ")
    # training configs
    parser.add_argument('--seed', type=int, default=111)#
    parser.add_argument('--print-freq', type=int, default=200)
    parser.add_argument('--eval-step', type=int, default=5)
    parser.add_argument('--temp', type=float, default=0.07)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    data_dir = "/home/zhiqi/dataset"
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default = data_dir)
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument("--cuda", type=str, default="2,3", help="cuda")
    
    main()
