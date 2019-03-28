#Main cell
import torch
from trainer import Trainer
from config import Config
from utils import prepare_dirs, save_config
from dataloader import get_test_loader, get_train_valid_loader
from dataloader import ToxicDataset
from torchvision import transforms
import numpy as np


# ensure directories are setup
config = Config()
prepare_dirs(config)


# ensure reproducibility
torch.manual_seed(config.random_seed)
kwargs = {}
if config.use_gpu:
    torch.cuda.manual_seed(config.random_seed)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    
# instantiate trainer
if (torch.cuda.is_available()):
    print("Cuda is available")


# instantiate data loaders
                    

# either train

if config.is_train:
    if config.cv:
        cv_folds = 5
        trans = transforms.Compose([transforms.ToTensor()])
        # load dataset
        dataset = ToxicDataset(csv_file="aggregate_tox.csv", root_dir=Config().data_dir)
        num_train = len(dataset)
        indices = list(range(num_train))
        values = []
        for fold in range(cv_folds):
            kwargs['fold'] = fold
            data_loader, indices = get_train_valid_loader(
                dataset, indices,config.data_dir, config.batch_size,
                config.random_seed, config.valid_size,
                config.shuffle, config.show_sample, config.cv,**kwargs
            )
            trainer = Trainer(config,data_loader)
            valid_acc = trainer.train()
            values.append(valid_acc)
            cross_file = open("cross_val.txt", "a+")
            cross_file.write(str(valid_acc))
            cross_file.write("\n")
            cross_file.close()
        cross_file = open("cross_val.txt", "a+")
        cross_file.write(str(np.mean(np.array(values))))
        cross_file.write("\n")
        cross_file.close()
    else:
        save_config(config)
        trainer.train()

# or load a pretrained model and test
else:
    data_loader = get_test_loader(config.data_dir, config.batch_size, **kwargs)
    trainer.test()
