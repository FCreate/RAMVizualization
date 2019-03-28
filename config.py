#Params of model
class Config(object):
      def __init__(self):
        #size of extracted patch at highest res
        self.patch_size =32

        #Scale of successive patches
        self.glimpse_scale = 2

        # # of downscaled patches per glimpse
        self.num_patches = 3

        #hidden size of loc fc
        self.loc_hidden = 512

        #hidden size of glimpse fc
        self.glimpse_hidden = 512

        # core network params
        ## of glimpses, i.e. BPTT iterations
        self.num_glimpses = 7

        #hidden size of rnn  
        self.hidden_size = 1024

        # reinforce params

        #gaussian policy standard deviation
        self.std = 0.17 #0.17
        #Monte Carlo sampling for valid and test sets
        self.M = 10

        # data params
        #Proportion of training set used for validation
        self.valid_size=0.2

        ## of images in each batch of data
        self.batch_size = 32

        ## of subprocesses to use for data loading
        self.num_workers = 4

        #Whether to shuffle the train and valid indices
        self.shuffle = True

        #Whether to visualize a sample grid of the data
        self.show_sample=False

        # training params

        #Whether to train or test the model
        self.is_train = True

        #Whether to train or test the model
        self.momentum=0.5

        ## of epochs to train for
        self.epochs = 500

        #Initial learning rate value
        self.init_lr = 3e-4

        #Number of epochs to wait before reducing lr
        self.lr_patience = 10

        #Number of epochs to wait before stopping train
        self.train_patience = 10

        # other params
        #Whether to run on the GPU
        self.use_gpu = True

        #Load best model or most recent for testing
        self.best = False
        #Seed to ensure reproducibility
        self.random_seed = 1

        #Directory in which data is stored
        self.data_dir = '../../RAM/Data/'

        #Directory in which to save model checkpoints
        self.ckpt_dir = './ckpt'
        #Directory in which Tensorboard logs wil be stored
        self.logs_dir='./logs/'
        #Whether to use tensorboard for visualization
        self.use_tensorboard= False
        #Whether to resume training from checkpoint
        self.resume = False

        #How frequently to print training details
        self.print_freq = 10
        #How frequently to plot glimpses
        self.plot_freq = 1
        
        self.cv = False
