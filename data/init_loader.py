import torch.utils.dataloader as dataloader
from data.frame_dataset import frame_dataset

def create_loader(args):
    dataset = frame_dataset(args)
    dataloader = dataloader(dataset, batch_size=args.batch_size,\
                                     shuffle=args.is_shuffle,\
                                     num_workers=args.n_threads)
    return dataloader