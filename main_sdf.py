import torch
import argparse

from sdf.utils import *
from time import time

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4, help="initial learning rate")
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")
    parser.add_argument('--save_grid', action='store_true', help="save grid")
    parser.add_argument('--marching_cubes_res', type=int, default=2048)

    opt = parser.parse_args()
    print(opt)

    seed_everything(opt.seed)

    if opt.ff:
        assert opt.fp16, "fully-fused mode must be used with fp16 mode"
        from sdf.network_ff import SDFNetwork
    elif opt.tcnn:
        assert opt.fp16, "tcnn mode must be used with fp16 mode"
        from sdf.network_tcnn import SDFNetwork        
    else:
        from sdf.network import SDFNetwork

    model = SDFNetwork(encoding="hashgrid")
    # model = SDFNetwork(encoding="hashgrid", num_layers=2, hidden_dim=256)
    print(model)
    # from sdf.provider import SDFDataset
    # from sdf.provider import SDF2Dataset as SDFDataset
    # train_dataset = SDFDataset(opt.path, size=100, num_samples=2**18)
    from sdf.provider import SDF3Dataset as SDFDataset
    train_dataset = SDFDataset(opt.path, size=305, num_samples=2**18)

    if opt.test:
        trainer = Trainer('ngp', model, workspace=opt.workspace, fp16=opt.fp16, 
               use_checkpoint='best', eval_interval=1,
               data_transform=train_dataset.transform,
               bounds_min=train_dataset.bounds_min,
               bounds_max=train_dataset.bounds_max,
               )
        trainer.save_mesh(os.path.join(opt.workspace, 'results', 'output.ply'), 
                1024, save_grid=opt.save_grid)

    else:
        from loss import mape_loss

        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0) #16)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)
        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

        valid_dataset = SDFDataset(opt.path, size=1, num_samples=2**18) # just a dummy
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)

        criterion = mape_loss # torch.nn.L1Loss()

        optimizer = lambda model: torch.optim.Adam([
            {'name': 'encoding', 'params': model.encoder.parameters()},
            {'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},
        ], lr=opt.lr, betas=(0.9, 0.99), eps=1e-15)

        # scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)

        trainer = Trainer('ngp', model, workspace=opt.workspace, optimizer=optimizer,
                criterion=criterion, ema_decay=0.95, fp16=opt.fp16, 
                lr_scheduler=scheduler, use_checkpoint='latest', eval_interval=1,
                data_transform=train_dataset.transform,
                bounds_min=train_dataset.bounds_min,
                bounds_max=train_dataset.bounds_max,
                path = opt.path,
                )

        trainer.train(train_loader, valid_loader, 20)

        # also test
        trainer.save_mesh(os.path.join(opt.workspace, 'results', 'output.ply'), opt.marching_cubes_res)
