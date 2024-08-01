from model import Generator_3 as Generator
from model import InterpLnr
import torch
import os
import time
import datetime
from collections import OrderedDict
from utils import quantize_f0_torch


class Solver(object):
    """Solver for training"""

    def __init__(self, data_loader, args, config):

        # Step configuration
        self.args = args
        self.num_iters = self.args.num_iters
        self.resume_iters = self.args.resume_iters
        self.log_step = self.args.log_step
        self.ckpt_save_step = self.args.ckpt_save_step
        self.return_latents = config.return_latents

        # Hyperparameters
        self.config = config

        # Data loader.
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)

        # Training configurations.
        self.lr = self.config.lr
        self.beta1 = self.config.beta1
        self.beta2 = self.config.beta2
        self.experiment = self.config.experiment
        self.bottleneck = self.config.bottleneck
        self.model_type = self.config.model_type
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device(
            'cuda:{}'.format(self.config.device_id) if self.use_cuda else 'cpu'
        )

        # Directories.
        self.model_save_dir = self.config.model_save_dir
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        # Build the model.
        self.build_model()

        # Logging
        self.min_loss_step = 0
        self.min_loss = float('inf')

    def build_model(self):
        self.model = Generator(self.config)
        self.print_network(self.model, self.model_type)
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

        self.Interp = InterpLnr(self.config)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.lr,
            [self.beta1, self.beta2],
            weight_decay=1e-6
        )
        self.Interp.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def print_optimizer(self, opt, name):
        print(opt)
        print(name)

    def restore_model(self, resume_iters):
        print(f"Loading the trained models from step {resume_iters}...")
        ckpt_name = "{}-{}-{}-{}.ckpt".format(
            self.experiment,
            self.bottleneck,
            self.model_type,
            resume_iters,
        )
        ckpt = torch.load(
            os.path.join(self.model_save_dir, ckpt_name),
            map_location=lambda storage, loc: storage
        )
        try:
            self.model.load_state_dict(ckpt['model'])
        except RuntimeError:
            new_state_dict = OrderedDict()
            for k, v in ckpt['model'].items():
                new_state_dict[k[7:]] = v
            self.model.load_state_dict(new_state_dict)
        self.lr = self.optimizer.param_groups[0]['lr']

    def train(self):
        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            print('Resuming ...')
            start_iters = self.resume_iters
            self.num_iters += self.resume_iters
            self.restore_model(self.resume_iters)
            self.print_optimizer(self.optimizer, 'optimizer')

        # Learning rate cache for decaying.
        lr = self.lr
        print('Current learning rates, lr: {}.'.format(lr))

        # Start training.
        print('Start training...')
        start_time = time.time()
        self.model = self.model.train()
        for i in range(start_iters, self.num_iters):

            # =============================================================== #
            #                   1. Load input data                            #
            # =============================================================== #

            print(f"loading data for epoch {i}")
            # Load data
            try:
                (
                    _,
                    spmel_gt,
                    rhythm_input,
                    content_input,
                    pitch_input,
                    timbre_input,
                    len_crop
                ) = next(self.data_iter)
            except:
                self.data_iter = iter(self.data_loader)
                (
                    _,
                    spmel_gt,
                    rhythm_input,
                    content_input,
                    pitch_input,
                    timbre_input,
                    len_crop
                ) = next(self.data_iter)

            # =============================================================== #
            #                    2. Train the model                           #
            # =============================================================== #

            print(f"training epoch {i}")

            # Move data to GPU if available
            spmel_gt = spmel_gt.to(self.device)
            rhythm_input = rhythm_input.to(self.device)
            content_input = content_input.to(self.device)
            pitch_input = pitch_input.to(self.device)
            timbre_input = timbre_input.to(self.device)
            len_crop = len_crop.to(self.device)

            # Prepare input data and apply random resampling
            content_pitch_input = torch.cat(
                (content_input, pitch_input), dim=-1)  # [B, T, F+1]
            content_pitch_input_intrp = self.Interp(
                content_pitch_input, len_crop)  # [B, T, F+1]
            pitch_input_intrp = quantize_f0_torch(
                content_pitch_input_intrp[:, :, -1])[0]  # [B, T, 257]
            content_pitch_input_intrp = torch.cat(
                # [B, T, F+257]
                (content_pitch_input_intrp[:, :, :-1], pitch_input_intrp),
                dim=-1
            )

            # Identity mapping loss
            if self.return_latents:
                (
                    spmel_output,
                    code_exp_1,
                    code_exp_2,
                    code_exp_3,
                    code_exp_4,
                ) = self.model(
                    content_pitch_input_intrp,
                    rhythm_input,
                    timbre_input,
                )
            else:
                spmel_output = self.model(
                    content_pitch_input_intrp,
                    rhythm_input,
                    timbre_input,
                )
            loss_id = torch.torch.nn.functional.mse_loss(
                spmel_output, spmel_gt)

            # Backward and optimize.
            loss = loss_id
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()

            # Logging.
            train_loss_id = loss_id.item()

            # ================================================================#
            #                           3. Logging and saving checkpoints     #
            # ================================================================#

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(
                    et, i+1, self.num_iters)
                log += ", {}/train_loss_id: {:.8f}".format(
                    self.model_type, train_loss_id)
                print(log)

            # Save model checkpoints
            if (i + 1) % self.ckpt_save_step == 0:
                ckpt_name = "{}-{}-{}-{}.ckpt".format(
                    self.experiment,
                    self.bottleneck,
                    self.model_type,
                    i + 1,
                )
                torch.save({
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, os.path.join(self.model_save_dir, ckpt_name))
                print(f"Saving model checkpoint into {self.model_save_dir}...")
