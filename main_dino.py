import os

import torch
import yaml
from torchvision import datasets
from data.multi_view_data_injector import MultiViewDataInjector
from data.transforms import get_simclr_data_transforms, get_simclr_data_transforms_onlyglobal
from models.mlp_head import MLPHead, MLPHead_DINO
from models.resnet_base_network import ResNet, MLPmixer, ResNet_BN_mom
from trainer_dino import BYOLTrainer
from CustomData import STL

print(torch.__version__)
torch.manual_seed(0)


def main():
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    data_transform = get_simclr_data_transforms(**config['data_transforms'])
    data_transform2 = get_simclr_data_transforms(**config['data_transforms'], blur=1.)
    # get_simclr_data_transforms_onlyglobal
    # data_transform = get_simclr_data_transforms_randAugment(config['data_transforms']['input_shape'])
    # data_transform2 = get_simclr_data_transforms_randAugment(config['data_transforms']['input_shape'])


    train_dataset = datasets.STL10('/media/snowflake/Data/', split='train+unlabeled', download=True,
                                   transform=MultiViewDataInjector([data_transform, data_transform2]))
    # train_dataset = STL(["/home/snowflake/Descargas/STL_data/unlabeled_images",
    #                      "/home/snowflake/Descargas/STL_data/train_images"],
    #                       transform=MultiViewDataInjector([data_transform, data_transform2]))



    # online network (the one that is trained)
    online_network = ResNet(**config['network']).to(device)
    # online_network = MLPmixer(**config['network']).to(device)

    # target encoder
    # target_network = ResNet_BN_mom(**config['network']).to(device)
    target_network = ResNet(**config['network']).to(device)
    # target_network = MLPmixer(**config['network']).to(device)

    pretrained_folder = config['network']['fine_tune_from']

    # load pre-trained model if defined
    if pretrained_folder:
        try:
            checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')

            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                                     map_location=torch.device(torch.device(device)))

            online_network.load_state_dict(load_params['online_network_state_dict'])
            target_network.load_state_dict(load_params['target_network_state_dict'])

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    # predictor network
    predictor = MLPHead_DINO(in_channels=online_network.projetion.net[-1].out_features,
                        **config['network']['projection_head']).to(device)


    optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),
                                **config['optimizer']['params'])

    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          predictor=predictor,
                          device=device,
                          **config['trainer'])

    trainer.train(train_dataset)


if __name__ == '__main__':
    main()