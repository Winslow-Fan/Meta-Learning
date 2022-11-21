import numpy as np
import torch
import numpy as py

class BatchSampler(object):
    def __init__(self, data, args, mode='train'):
        super(BatchSampler, self).__init__()
        self.label = data.data.label

        if mode is 'train':
            self.cls_iter = args.cls_iter_tr
            self.sample = args.supp_tr + args.query_tr
        elif mode is 'val':
            self.cls_iter = args.cls_iter_val
            self.sample = args.supp_val + args.query_val

        self.iterations = args.iters
        self.classes, self.counts = np.unique(self.label, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        self.idx = range(len(self.label))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan

        self.numel_cls = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.label):
            label_idx = np.argwhere(self.classes==label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_cls[label_idx] += 1

    def __iter__(self):
        spc = self.sample
        cpi = self.cls_iter

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]

            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i*spc, (i+1)*spc)
                label_idx = torch.arange(len(self.classes)).long()[self.classes==c].item()
                sample_idxs = torch.randperm(self.numel_cls[label_idx])[:spc]
                # print(type(label_idx), type(sample_idxs))
                # print(type(self.indexes[label_idx][sample_idxs]))
                batch[s] = torch.LongTensor(self.indexes[label_idx][sample_idxs])


            batch = batch[torch.randperm(len(batch))]
            yield batch
