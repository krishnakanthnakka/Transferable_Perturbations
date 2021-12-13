from .base_options1 import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_port', type=int,
                            default=8097, help='visdom port of the web display')

        parser.add_argument('--display_freq', type=int, default=100,
                            help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4,
                            help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1,
                            help='window id of the web display')
        parser.add_argument('--display_server', type=str,
                            default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main',
                            help='visdom display environment name (default is "main")')
        parser.add_argument('--update_html_freq', type=int, default=1000,
                            help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=30,
                            help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true',
                            help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000,
                            help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=10,
                            help='frequency of saving checkpoints at the end of epochs')
        # parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true',
                            help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true',
                            help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str,
                            default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--niter', type=int, default=100,
                            help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100,
                            help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5,
                            help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002,
                            help='initial learning rate for adam')
        parser.add_argument('--lr_gamma', type=float,
                            default=0.3, help='gamma decay rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan',
                            help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=100,
                            help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='step',
                            help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=25,
                            help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--attackobjective', type=str,
                            default='Targeted', help=' attack type Blind| targeted')
        parser.add_argument('--log', type=str, default=print, help=' logger')

        parser.add_argument('--eps', type=float, default=16 /
                            255.0, help='initial learning rate for adam')
        parser.add_argument('--weight_L2', type=float,
                            default=100, help='initial learning rate for adam')
        parser.add_argument('--weight_ce', type=float,
                            default=100, help='initial learning rate for adam')
        parser.add_argument('--weight_rl', type=float, default=1,
                            help='initial learning rate for adam')
        parser.add_argument('--weight_att', type=float, default=2,
                            help='initial learning rate for adam')
        parser.add_argument('--detcfg', type=str)
        parser.add_argument('--weight_feat', type=float, default=2,
                            help='initial learning rate for adam')
        parser.add_argument('--cls_margin', type=float, default=-5,
                            help='initial learning rate for adam')
        parser.add_argument('--run', type=str)
        parser.add_argument('--eval_dataset', type=str)
        parser.add_argument('--detckpt', type=str)
        parser.add_argument('--cls_thres', type=float)
        parser.add_argument('--perturbmode', type=int)
        parser.add_argument('--train_classifier', type=str)
        parser.add_argument('--loss_type', type=str)
        parser.add_argument('--train_dataset', type=str)
        parser.add_argument('--pretrained_netG', action='store_true')
        parser.add_argument('--detmodel', type=str)
        parser.add_argument('--eval_num_classes', type=int)
        parser.add_argument('--eval_model', type=str)
        parser.add_argument('--load_epoch', type=int)
        parser.add_argument('--train_config', type=str)
        parser.add_argument('--att_order', type=int)
        parser.add_argument('--max_epochs', type=int)
        parser.add_argument('--pretrain_weights', type=str)
        parser.add_argument('--netg_depth', type=str)
        parser.add_argument('--num_images', type=int)
        parser.add_argument('--data_dim', type=str)
        parser.add_argument('--gen_dropout', type=float)
        parser.add_argument('--act_layer', type=int, nargs="*", default=[-1, -1])
        parser.add_argument('--softmax2D', action='store_true')
        parser.add_argument('--loss_fn', type=str)
        parser.add_argument('--data_shuffle', action='store_true')
        parser.add_argument('--act_layer_mean', action='store_true')
        parser.add_argument('--data_aug', type=str, default='None')
        parser.add_argument('--classifier_weights', type=str)
        parser.add_argument('--train_getG_299', action='store_true')
        parser.add_argument('--smoothing', action='store_true')
        parser.add_argument('--train_classifier_weights', type=str)
        parser.add_argument('--train_classifier2', type=str)
        parser.add_argument('--act_layer2', type=int, nargs="*", default=[-1, -1])
        parser.add_argument('--warm_start', action='store_true')
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--warm_start_L2_steps', type=int, default=-5)
        parser.add_argument('--defense', type=str, default="")

        self.isTrain = True
        return parser
