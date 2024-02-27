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
    parser.add_argument('--scene_list', type=list, default=["7"])

    opt = parser.parse_args()
    # scene_list = ["c49a8c6cff"]
    # scene_list = ["c49a8c6cff", "785e7504b9"]
    scene_list = []
    if "7" in opt.scene_list:
        scene_list += ["785e7504b9"]
    if "c" in opt.scene_list:
        scene_list += ["c49a8c6cff"]
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
        # from sdf.network import SDF2Network as SDFNetwork

    model = SDFNetwork(encoding="hashgrid", n_encoders=len(scene_list))
    # model = SDFNetwork(encoding="hashgrid", num_layers=2, hidden_dim=256, n_encoders=len(scene_list))
    print(model)
    # from sdf.provider import SDFDataset
    # from sdf.provider import SDF2Dataset as SDFDataset
    # train_dataset = SDFDataset(opt.path, size=100, num_samples=2**18)
    from sdf.provider import SDF5Dataset as SDFDataset
    train_dataset = SDFDataset(opt.path, size=305, num_samples=2**18, scene_list=scene_list, dummy=opt.test)
    # train_dataset = SDFDataset(opt.path, size=305, num_samples=2**18, scene_list=scene_list, dummy=opt.test)
    # train_dataset = SDFDataset(opt.path, size=1, num_samples=2**18, scene_list=scene_list)

    if opt.test:
        trainer = Trainer('ngp', model, workspace=opt.workspace, fp16=opt.fp16, 
               use_checkpoint='best', eval_interval=1,
               data_transform=train_dataset.transform,
               bounds_min=train_dataset.bounds_min,
               bounds_max=train_dataset.bounds_max,
               path = opt.path,
               )
        # path = 
        # mesh = trimesh.load(path, preprocess=False)
        pts = torch.from_numpy(train_dataset.pts).float()
        trainer.model.encoders[0] = trainer.model.encoder
        feats = trainer.get_point_features(pts, 0)
        feats = feats.cpu().numpy().astype(np.float16)
        pause()
        with open(opt.path[:-4] + "1.npy", "wb") as f:
            np.save(f, feats)
        exit()
        trainer.save_mesh(os.path.join(opt.workspace, 'results', 'output.ply'), 
                1024, save_grid=opt.save_grid)

    else:
        from loss import mape_loss

        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0) #16)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)
        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

        valid_dataset = SDFDataset(opt.path, size=1, num_samples=2**18, scene_list=scene_list) # just a dummy
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)

        criterion = mape_loss # torch.nn.L1Loss()
        optimizer = lambda model: torch.optim.Adam([
            {'name': 'encoding'+str(i), 'params': model.encoders[i].parameters()} for i in range(len(model.encoders))]+
            [{'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},
        ], lr=opt.lr, betas=(0.9, 0.99), eps=1e-15)

        # optimizer = lambda model: torch.optim.Adam([
            # {'name': 'encoding', 'params': model.encoder.parameters()},
            # {'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},
        # ], lr=opt.lr, betas=(0.9, 0.99), eps=1e-15)

        # scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
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
        for i, scene in enumerate(scene_list): 
            # if i==0: continue
            trainer.save_mesh(os.path.join(opt.workspace, 'results', 'output.ply'), opt.marching_cubes_res, encoder_id=i, scene=scene)
