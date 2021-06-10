import os
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import math
from utils import _create_model_training_folder
from numpy import array

from torch.optim.swa_utils import AveragedModel, SWALR

class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, device, **params):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.max_epochs = params['max_epochs']
        self.writer = SummaryWriter()
        self.m = params['m']
        self.m_initial = params['m']
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']
        _create_model_training_folder(self.writer, files_to_same=["./config/config.yaml", "main.py", 'trainer.py'])
        self.center = None
        self.bn = torch.nn.BatchNorm1d(256, affine=False).cuda()

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1) # L2 loss, similar to cosine similarity


    @staticmethod
    def regression_loss_with_negatives(x, y, tau = 1.0):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        logits = torch.mm(x, y.t())
        labels = torch.tensor(array(range(logits.shape[0]))).cuda()
        ce_loss = torch.nn.CrossEntropyLoss()
        loss = ce_loss(logits/tau, labels)
        return 2 * tau * loss

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_epochs, eta_min=0.0005, last_epoch=-1)

        swa_start = 200

        niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.initializes_target_network()
        self.optimizer.zero_grad()

        for epoch_counter in range(self.max_epochs):
            for (batch_view_1, batch_view_2), _ in train_loader:
                # print(batch_view_1.shape) # 256, 3, 96, 96
                batch_view_1 = batch_view_1.to(self.device)
                batch_view_2 = batch_view_2.to(self.device)

                loss = self.update(batch_view_1, batch_view_2)
                self.writer.add_scalar('loss', loss, global_step=niter)

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
                self._update_target_network_parameters()  # update the key encoder

                niter += 1

            print("End of epoch {}".format(epoch_counter))
            scheduler.step()
            self.m = 1 - (1 - self.m_initial) * (math.cos(math.pi * epoch_counter / self.max_epochs) + 1) / 2
            self.save_model(os.path.join(model_checkpoints_folder, 'model.pth'))


    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def barlow_loss(self, x, y):
        c = self.bn(x).T @ self.bn(y)
        c = c / self.batch_size
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(0.03125)
        off_diag = self.off_diagonal(c).pow_(2).sum().mul(0.03125)
        loss = on_diag + 3.9e-3 * off_diag
        return loss


    # def update(self, batch_view_1, batch_view_2):
    #     # compute query feature
    #
    #     predictions_from_view_1 = self.online_network(batch_view_1)
    #     predictions_from_view_2 = self.online_network(batch_view_2)
    #
    #     # # compute key features
    #     # with torch.no_grad():  # teacher/target netowrk
    #     #     targets_to_view_2 = self.target_network(batch_view_1).detach()
    #     #     targets_to_view_1 = self.target_network(batch_view_2).detach()
    #
    #     loss = self.barlow_loss(predictions_from_view_1, predictions_from_view_2)
    #     # loss += self.barlow_loss(predictions_from_view_2, targets_to_view_2)
    #
    #     return loss.mean()

    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))

        # compute key features
        with torch.no_grad():  # teacher/target netowrk
            targets_to_view_2 = self.target_network(batch_view_1).detach()
            targets_to_view_1 = self.target_network(batch_view_2).detach()

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)

        # loss = self.regression_loss_with_negatives(predictions_from_view_1, targets_to_view_1)
        # loss += self.regression_loss_with_negatives(predictions_from_view_2, targets_to_view_2)

        return loss.mean()

    def save_model(self, PATH, save_swa=False, swa=None):
        print(PATH)
        if save_swa:
            self.online_network = swa

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)
