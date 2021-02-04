import os, gc, time
import numpy as np
import torch
import importlib

import torchvision.transforms as transforms
import torchvision.datasets as datasets

try:
    from nvidia import dali
    from dali import HybridTrainPipe, HybridValPipe, DaliIteratorCPU, DaliIteratorGPU
except:
    print('Could not import DALI')

def clear_memory(verbose=False):
    stt = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # https://forums.fast.ai/t/clearing-gpu-memory-pytorch/14637
    gc.collect()

    if verbose:
        print('Cleared memory.  Time taken was %f secs' % (time.time() - stt))


class Dataset():
    """
    Pytorch Dataloader, with torchvision or Nvidia DALI CPU/GPU pipelines.


    data_dir (str): Directory to dataset.  Format should be the same as torchvision dataloader,
    batch_size (int): how many samples per batch to load
    size (int): Output size (typically 224 for ImageNet)
    val_size (int): Validation pipeline resize size (typically 256 for ImageNet)
    workers (int): how many workers to use for data loading
    cuda (bool): Output tensors on CUDA, CPU otherwise
    use_dali (bool): Use Nvidia DALI backend, torchvision otherwise
    dali_cpu (bool): Use Nvidia DALI cpu backend, GPU backend otherwise
    fp16 (bool, optional, default = False) - Output the data in fp16 instead of fp32
    mean (tuple): Image mean value for each channel
    std (tuple): Image standard deviation value for each channel
    pin_memory_dali (bool): Transfer CPU tensor to pinned memory before transfer to GPU (dali only)
    """

    def __init__(self,
                 data_dir,
                 batch_size,
                 size=224,
                 val_batch_size=1, # Check what is the story with this?
                 val_size=None,
                 workers=4,
                 cuda=True,
                 use_dali=True,
                 dali_cpu=False,
                 fp16=True,
                 mean=(0.5,0.5,0.5),
                 std=(0.5,0.5,0.5),
                 pin_memory_dali=True,
                 ):

            self.batch_size = batch_size
            self.size = size
            self.val_batch_size = val_batch_size
            self.workers = workers
            self.cuda = cuda
            self.use_dali = use_dali
            self.dali_cpu = dali_cpu
            self.fp16 = fp16
            self.mean = mean
            self.std = std
            self.pin_memory_dali = pin_memory_dali

            self.val_size = val_size
            if self.val_size is None:
                self.val_size = self.size

            if self.val_batch_size is None:
                self.val_batch_size = self.batch_size

            # Data loading code
            self.traindir = os.path.join(data_dir, 'train')
            self.valdir = os.path.join(data_dir, 'val')

            print(self.traindir)
            print('Using Nvidia DALI dataloader')
            assert len(datasets.ImageFolder(self.valdir)) % self.val_batch_size == 0, 'Validation batch size must divide validation dataset size cleanly...  DALI has problems otherwise.'
            self._build_dali_pipeline()



    def _build_dali_pipeline(self, val_on_cpu=True):

        iterator_train = DaliIteratorGPU

        self.train_pipe = HybridTrainPipe(batch_size=self.batch_size, num_threads=self.workers, device_id=0,
                                          data_dir=self.traindir, crop=self.size, dali_cpu=self.dali_cpu,
                                          mean=self.mean, std=self.std, local_rank=0,
                                          world_size=self.world_size, shuffle=True, fp16=self.fp16, min_crop_size=self.min_crop_size)

        self.train_pipe.build()
        self.train_loader = iterator_train(pipelines=self.train_pipe, size=self.get_nb_train() / self.world_size, fp16=self.fp16, mean=self.mean, std=self.std, pin_memory=self.pin_memory_dali)

        iterator_val = DaliIteratorGPU
        if val_on_cpu:
            iterator_val = DaliIteratorCPU

        self.val_pipe = HybridValPipe(batch_size=self.val_batch_size, num_threads=self.workers, device_id=0,
                                      data_dir=self.valdir, crop=self.size, size=self.val_size, dali_cpu=val_on_cpu,
                                      mean=self.mean, std=self.std, local_rank=0,
                                      world_size=self.world_size, shuffle=False, fp16=self.fp16)

        self.val_pipe.build()
        self.val_loader = iterator_val(pipelines=self.val_pipe, size=self.get_nb_val() / self.world_size, fp16=self.fp16, mean=self.mean, std=self.std, pin_memory=self.pin_memory_dali)


    def get_train_loader(self):
        """
        Creates & returns an iterator for the training dataset
        :return: Dataset iterator object
        """
        if self.use_dali:
            return self.train_loader
        return self._get_torchvision_loader(loader=self.train_loader)

    def get_val_loader(self):
        """
        Creates & returns an iterator for the training dataset
        :return: Dataset iterator object
        """
        if self.use_dali:
            return self.val_loader
        return self._get_torchvision_loader(loader=self.val_loader)

    def get_nb_train(self):
        """
        :return: Number of training examples
        """
        if self.use_dali:
            return int(self.train_pipe.epoch_size("Reader"))
        return len(datasets.ImageFolder(self.traindir))

    def get_nb_val(self):
        """
        :return: Number of validation examples
        """
        if self.use_dali:
            return int(self.val_pipe.epoch_size("Reader"))
        return len(datasets.ImageFolder(self.valdir))

    def prep_for_val(self):
        self.reset(val_on_cpu=False)

    # This is needed only for DALI
    def reset(self, val_on_cpu=True):
        if self.use_dali:
            clear_memory()

            # Currently we need to delete & rebuild the dali pipeline every epoch,
            # due to a memory leak somewhere in DALI
            print('Recreating DALI dataloaders to reduce memory usage')
            del self.train_loader, self.val_loader, self.train_pipe, self.val_pipe
            clear_memory()

            # taken from: https://stackoverflow.com/questions/1254370/reimport-a-module-in-python-while-interactive
            importlib.reload(dali)
            from dali import HybridTrainPipe, HybridValPipe, DaliIteratorCPU, DaliIteratorGPU

            self._build_dali_pipeline(val_on_cpu=val_on_cpu)

    def set_train_batch_size(self, train_batch_size):
        self.batch_size = train_batch_size
        if self.use_dali:
            del self.train_loader, self.val_loader, self.train_pipe, self.val_pipe
            self._build_dali_pipeline()
        else:
            del self.train_sampler, self.val_sampler, self.train_loader, self.val_loader
            self._build_torchvision_pipeline()

    def get_nb_classes(self):
        """
        :return: The number of classes in the dataset - as indicated by the validation dataset
        """
        return len(datasets.ImageFolder(self.valdir).classes)


def fast_collate(batch):
    """Convert batch into tuple of X and Y tensors."""
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets
