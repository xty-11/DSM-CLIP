import torchvision.transforms as T
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np
import torch.nn as nn
from ..common import CommDataset

def val_collate_fn(batch):
    imgs, pids, camids, viewids, image_path, domains, cid = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    cid = torch.tensor(cid, dtype=torch.int64)
    domains = torch.tensor(domains, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, domains


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)  # dict with list value
        # {783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, data in enumerate(self.data_source):
            self.index_dic[data[1]].append(index)
        self.pids = list(self.index_dic.keys())
        print('pids', len(self.pids))
        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        pids = self.pids
        for pid in pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
        print(len(final_idxs))
        return iter(final_idxs)

    def __len__(self):
        return self.length


class DHS(Sampler):
    def __init__(self, data_source, batch_size, num_instances, model,pid2label,h_size=224, w_size=224,epoch=60,lambda_text=1.0):
        super().__init__(data_source)
        self.lambda_text = lambda_text
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.model = model
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(dict)

        for index, data in enumerate(self.data_source):
            if data[2] in self.index_dic[data[1]].keys():
                self.index_dic[data[1]][data[2]].append(index)
            else:
                self.index_dic[data[1]][data[2]] = [index]
        self.pids = list(self.index_dic.keys())
        self.pidlabel = torch.tensor([pid2label[pid] for pid in self.pids])
        self.dist_mat = None
        self.length = 0
        self.epoch = epoch
        self.num_epoch = 0
        for pid in self.pids:
            num = sum([len(self.index_dic[pid][key]) for key in self.index_dic[pid].keys()])
            self.length += num

        self.eval_transforms = T.Compose([
            T.Resize((h_size, w_size)),
            T.ToTensor(),
            nn.InstanceNorm2d(3),
        ])

    # sort for camera
    def sort_dic_cam(self, s):
        ks = list(s.keys())
        len_k = np.array([len(s[k]) for k in s.keys()])
        ix = len_k.argsort()[::-1]
        return {ks[i]: s[ks[i]] for i in ix}

    def __iter__(self):
        self.num_epoch+=1
        batch_idxs_dict = defaultdict(list)
        pids = copy.deepcopy(self.pids)
        random.shuffle(pids)
        print("Start by generating batches by camera...")
        for pid in pids:
            dic_tmp = copy.deepcopy(self.index_dic[pid])
            cids = list(dic_tmp.keys())
            for cid in cids:
                random.shuffle(dic_tmp[cid])
            idxs = []
            while cids:
                num = 0
                dic_tmp = self.sort_dic_cam(dic_tmp)
                for cid in cids:
                    num += 1
                    idxs.append(dic_tmp[cid].pop())
                    if len(dic_tmp[cid]) == 0:
                        cids.remove(cid)
                    if num == self.num_instances:
                        break
            if len(idxs) <= 1:
                continue
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)

            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        print("Hard negative samples are generated according to the distance...")
        final_idxs = []
        model = copy.deepcopy(self.model).cuda().eval()
        index_dic = defaultdict(list)
        for index, data in enumerate(self.data_source):
            index_dic[data[1]].append(index)
        pids = list(index_dic.keys())
        inex_dic = {k: index_dic[k][random.randint(0, len(index_dic[k]) - 1)] for k in pids}
        choice_set = CommDataset([self.data_source[i] for i in list(inex_dic.values())], self.eval_transforms,
                                 relabel=True)
        choice_loader = DataLoader(
            choice_set, batch_size=128, shuffle=False, num_workers=8,
            collate_fn=val_collate_fn
        )

        feats = torch.tensor([]).cuda()
        feats_text = torch.tensor([]).cuda()
        for i, (img, pid, _) in enumerate(choice_loader):
            with torch.no_grad():
                img = img.cuda()
                feat = model(img)
                feats = torch.cat((feats, feat), dim=0)
                label_text = self.pidlabel[pid]
                feat_text = model(label=label_text, get_text=True)
                feats_text = torch.cat((feats_text, feat_text), dim=0)
        dist_mat_text = euclidean_dist(feats_text, feats_text)

        # Calculate the dual-modal similarity
        dist_mat = euclidean_dist(feats, feats) + self.lambda_text*dist_mat_text  
        
        for i in range(len(dist_mat)):
            dist_mat[i][i] = float("inf")
        feat_dist = {}
        for i, feat in enumerate(dist_mat):
            loc = torch.argsort(feat)
            feat_dist[pids[i]] = [pids[int(loc[j].cpu())] for j in range(31)]

        random.shuffle(pids)   # add batch randomness
        i = 0
        for k in pids:
            if i > 24320:  # hard negative samples are enough to break out of the loop early
                break
            i += 1
            v = feat_dist[k]
            final_idxs.extend(batch_idxs_dict[k][random.randint(0, len(batch_idxs_dict[k]) - 1)])
            for k in v:
                i += 1
                final_idxs.extend(batch_idxs_dict[k][random.randint(0, len(batch_idxs_dict[k]) - 1)])
        self.length = len(final_idxs)
        print(self.length)
        return iter(final_idxs)

    def __len__(self):
        return self.length


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist









class GS(Sampler):
    def __init__(self, data_source, batch_size, num_instances, model):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.model = model
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(dict)
        self.index_dic_imgpath = defaultdict(list)

        for index, data in enumerate(self.data_source):
            if data[2] in self.index_dic[data[1]].keys():
                self.index_dic[data[1]][data[2]].append(index)
            else:
                self.index_dic[data[1]][data[2]] = [index]
            self.index_dic_imgpath[data[1]].append(data[0])
            self.domain_info[data[-1].lower()] += 1
        self.pids = list(self.index_dic.keys())

        self.length = 0
        for pid in self.pids:
            num = sum([len(self.index_dic[pid][key]) for key in self.index_dic[pid].keys()])
            self.length += num

    def sort_dic(self, s):
        ks = list(s.keys())
        len_k = np.array([len(s[k]) for k in s.keys()])
        ix = len_k.argsort()[::-1]
        return {ks[i]: s[ks[i]] for i in ix}

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        pids = copy.deepcopy(self.pids)
        random.shuffle(pids)
        print("Dist Updating!")
        for pid in pids:
            dic_tmp = copy.deepcopy(self.index_dic[pid])

            cids = list(dic_tmp.keys())
            for cid in cids:
                random.shuffle(dic_tmp[cid])
            idxs = []
            while cids:
                num = 0
                dic_tmp = self.sort_dic(dic_tmp)
                for cid in cids:
                    num += 1
                    idxs.append(dic_tmp[cid].pop())
                    if len(dic_tmp[cid]) == 0:
                        cids.remove(cid)
                    if num == self.num_instances:
                        break
            if len(idxs) <= 1:
                continue
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)

            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        final_idxs = []
        transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

        model = copy.deepcopy(self.model).cuda().eval()
        index_dic = defaultdict(list)
        for index, data in enumerate(self.data_source):
            index_dic[data[1]].append(index)
        pids = list(index_dic.keys())
        inex_dic = {k: index_dic[k][random.randint(0, len(index_dic[k]) - 1)] for k in pids}
        feat_dist = {}
        choice_set = CommDataset([self.data_source[i] for i in list(inex_dic.values())], transforms, relabel=False)
        choice_loader = DataLoader(
            choice_set, batch_size=256, shuffle=False, num_workers=8,
            collate_fn=val_collate_fn
        )
        feats = torch.tensor([]).cuda()
        for i, (img, _, _) in enumerate(choice_loader):
            with torch.no_grad():
                img = img.cuda()
                feat = model(img)
                feats = torch.cat((feats, feat), dim=0)

        dist_mat = euclidean_dist(feats, feats)
        for i in range(len(dist_mat)):
            dist_mat[i][i] = float("inf")

        for i, feat in enumerate(dist_mat):
            loc = torch.argsort(feat)
            feat_dist[pids[i]] = [pids[int(loc[j].cpu())] for j in range(31)]

        random.shuffle(pids)
        i = 0
        for k in pids:
            if i > 24320:
                break
            i += 1
            v = feat_dist[k]
            final_idxs.extend(batch_idxs_dict[k][random.randint(0, len(batch_idxs_dict[k]) - 1)])
            for k in v:
                i += 1
                final_idxs.extend(batch_idxs_dict[k][random.randint(0, len(batch_idxs_dict[k]) - 1)])
        self.length = len(final_idxs)
        print(self.length)
        return iter(final_idxs)

    def __len__(self):
        return self.length

