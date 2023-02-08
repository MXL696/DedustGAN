import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
from util import util
from PIL import Image
from torchvision.transforms import ToTensor



class DedustModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_G', type=float, default=0.2, help='weight for loss_G_single')
            parser.add_argument('--lambda_identity', type=float, default=1,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_rec_I', type=float, default=1, help='weight for loss_rec_I')

        parser.add_argument('--G_L', type=str, default='unet_trans_256', help='specify generator architecture')
        parser.add_argument('--G_J', type=str, default='resnet_9blocks', help='specify generator architecture')


        return parser

    def __init__(self, opt):
        """Initialize the RefineDNet class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_single', 'G_single', 'rec_I', 'idt_J']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>

        if self.isTrain:
            self.visual_names = ['real_I', 'Lt', 'refine_J', 'rec_I',
                                 'rec_J', 'real_J', 'ref_real_J']
        else:
            self.visual_names = ['real_I', 'refine_J', 'rec_I', 'rec_J']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_L', 'G_J', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G_L', 'G_J']

        # define networks (both Generators and discriminators)
        self.G_L = networks.define_G(3, 3, opt.ngf, opt.G_L, opt.norm,
                                              not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.G_J = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.G_J, opt.norm,
                                              not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.D = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert (opt.input_nc == opt.output_nc)
            self.fake_I_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_J_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionRec = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.G_L.parameters(), self.G_J.parameters()), lr=opt.lr,
                betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)



    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.real_I = input['dusty'].to(self.device)  # [-1, 1]
        self.image_paths = input['paths']

        if self.isTrain:
            self.real_J = input['clear'].to(self.device)  # [-1, 1]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        self.Lt = self.G_L(self.real_I)
        self.refine_J = self.G_J(self.real_I)
        self.rec_I = util.synthesize(self.refine_J, self.Lt)
        self.rec_J = util.reverse(self.real_I, self.Lt)

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()



    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):
        fake_J = self.fake_I_pool.query(self.refine_J)
        self.loss_D_single = self.backward_D_basic(self.D, self.real_J, fake_J)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_G = self.opt.lambda_G
        lambda_rec_I = self.opt.lambda_rec_I


        # Generator losses for rec_I and refine_J
        self.loss_G_single = self.criterionGAN(self.D(self.refine_J), True) * lambda_G

        # Reconstrcut loss
        self.loss_rec_I = self.criterionRec(self.real_I, self.rec_I) * lambda_rec_I


        # Identity loss
        self.ref_real_J = self.G_J(self.real_J)
        self.loss_idt_J = self.criterionIdt(self.ref_real_J, self.real_J) * lambda_idt

        self.loss_G = self.loss_G_single + self.loss_rec_I + self.loss_idt_J

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad(self.D, False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad(self.D, True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D()  # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights
