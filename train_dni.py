# -*- coding: utf-8 -*-


import os
import argparse
import time
from datetime import datetime
import numpy as np
import json

import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler

# zcr lib
from prednet import PredNet
from prednet_dni import PredNetDNI
from data_utils import ZcrDataLoader

# Import evaluate
from evaluate_dni import evaluate

# Sagemaker deployment functions
from deploy import model_fn
from deploy import input_fn
from deploy import predict_fn

model_fn = model_fn
input_fn = input_fn
predict_fn = predict_fn


# os.environ['CUDA_LAUNCH_BLOCKING'] = 1
# torch.backends.cudnn.benchmark = True


def arg_parse():
    desc = "Video Frames Predicting Task via PredNet."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--mode", default="train", type=str, help="train or evaluate (default: train)"
    )
    parser.add_argument(
        "--evaluate",
        default=True,
        type=bool,
        help="evaluate after training. (default: true)",
    )
    parser.add_argument(
        "--epochs",
        default=2,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--non-local",
        default=True,
        type=bool,
        help="Indicates that the model files should be downloaded from a remote repo.",
    )
    parser.add_argument(
        "--num_plot", default=40, type=int, metavar="N", help="how many images to plot"
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, metavar="N", help="The size of batch"
    )
    parser.add_argument(
        "--optimizer", default="SGD", type=str, help="which optimizer to use"
    )
    parser.add_argument(
        "--lr", default=0.001, type=float, metavar="LR", help="initial learning rate"
    )
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum for SGD")
    parser.add_argument(
        "--beta1", default=0.9, type=float, help="beta1 in Adam optimizer"
    )
    parser.add_argument(
        "--beta2", default=0.99, type=float, help="beta2 in Adam optimizer"
    )
    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--printCircle",
        default=10,
        type=int,
        metavar="N",
        help="how many steps to print the loss information",
    )
    parser.add_argument(
        "--data_format",
        default="channels_last",
        type=str,
        help="(c, h, w) or (h, w, c)?",
    )
    parser.add_argument(
        "--n_channels",
        default=3,
        type=int,
        metavar="N",
        help="The number of input channels (default: 3)",
    )
    parser.add_argument(
        "--img_height",
        default=128,
        type=int,
        metavar="N",
        help="The height of input frame (default: 128)",
    )
    parser.add_argument(
        "--img_width",
        default=160,
        type=int,
        metavar="N",
        help="The width of input frame (default: 160)",
    )
    parser.add_argument(
        "--layer_loss_weightsMode",
        default="L_0",
        type=str,
        help="L_0 or L_all for loss weights in PredNet",
    )
    parser.add_argument(
        "--num_timeSteps",
        default=10,
        type=int,
        metavar="N",
        help="number of timesteps used for sequences in training (default: 10)",
    )
    parser.add_argument(
        "--step",
        default=1,
        type=int,
        help="The step size for image sequences (default: 1)",
    )
    parser.add_argument("--shuffle", default=True, type=bool, help="shuffle or not")
    parser.add_argument(
        "--training-data-dir",
        default="h5/",
        type=str,
        help="Training data directory for the training and validation datasets.",
    )
    parser.add_argument(
        "--load-model",
        default="",
        type=str,
        help="Path to pre-existing model that can be loaded before training.",
    )
    parser.add_argument(
        "--seed", default=1234, type=int, help="Random seed for training.",
    )

    # Container environment
    parser.add_argument(
        "--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"])
    )
    parser.add_argument(
        "--current-host", type=str, default=os.environ["SM_CURRENT_HOST"]
    )
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--data-dir", type=str, default="/opt/ml/input/data/h5")
    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )

    args = parser.parse_args()
    return args


def print_args(args):
    print("-" * 50)
    for arg, content in args.__dict__.items():
        print("{}: {}".format(arg, content))
    print("-" * 50)


def train(model, args):
    """Train PredNet on KITTI sequences"""

    # print('layer_loss_weightsMode: ', args.layer_loss_weightsMode)
    prednet_dni = model
    # frame data files
    training_data_dir = args.data_dir
    output_data_dir = args.output_data_dir
    train_file = os.path.join(training_data_dir, "train.h5")
    train_sources = os.path.join(training_data_dir, "sources_train.h5")

    output_mode = "error"
    sequence_start_mode = "all"
    N_seq = None
    dataLoader = ZcrDataLoader(
        train_file, train_sources, output_mode, sequence_start_mode, N_seq, args
    ).dataLoader()
    input_shape = (args.batch_size, args.num_timeSteps, n_channels, img_height, img_width)

    optimizer = torch.optim.Adam(prednet_dni.parameters(), lr=args.lr)
    # This is not the same LR scheduler as the original paper but supports loss observations
    lr_maker = lr_scheduler.StepLR(optimizer=optimizer, step_size=3000, gamma=0.1)
    printCircle = args.printCircle
    for e in range(args.epochs):
        tr_loss = 0.0
        sum_trainLoss_in_epoch = 0.0
        min_trainLoss_in_epoch = float("inf")
        startTime_epoch = time.time()

        initial_states_dni = prednet_dni.get_initial_states(
            input_shape
        )  # 原网络貌似不是stateful的, 故这里再每个epoch开始时重新初始化(如果是stateful的, 则只在全部的epoch开始时初始化一次)
        states = initial_states_dni
        for step, (frameGroup, target) in enumerate(dataLoader):
            #             print(frameGroup.size())   # [torch.FloatTensor of size 16x12x80x80]
            batch_frames = frameGroup.cuda()
            _input = prednet_dni(batch_frames, states)
            
            # Use last DNI measurement
            target = target[-1].cuda()

            loss = torch.nn.MSELoss()
            loss = loss(_input, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_maker.step()

            tr_loss += loss.item()
            sum_trainLoss_in_epoch += loss.item()
            if step % printCircle == (printCircle - 1):
                print(
                    "epoch: [%3d/%3d] | [%4d/%4d]  loss: %.4f  lr: %.5lf"
                    % (
                        (e + 1),
                        args.epochs,
                        (step + 1),
                        len(dataLoader),
                        tr_loss / printCircle,
                        optimizer.param_groups[0]["lr"],
                    )
                )
                tr_loss = 0.0

        endTime_epoch = time.time()
        print(
            "Time Consumed within an epoch: %.2f (s)"
            % (endTime_epoch - startTime_epoch)
        )

        if sum_trainLoss_in_epoch < min_trainLoss_in_epoch:
            min_trainLoss_in_epoch = sum_trainLoss_in_epoch
            zcr_state_dict = {
                "epoch": (e + 1),
                "tr_loss": min_trainLoss_in_epoch,
                "state_dict": prednet_dni.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            saveCheckpoint(zcr_state_dict, output_data_dir)


def saveCheckpoint(zcr_state_dict, output_data_dir):
    """save the checkpoint for both restarting and evaluating."""
    epoch = zcr_state_dict["epoch"]
    fileName = f"checkpoint-{epoch}"
    path = os.path.join(output_data_dir, fileName)
    torch.save(zcr_state_dict, path)


if __name__ == "__main__":
    args = arg_parse()
    print_args(args)

    n_channels = args.n_channels
    img_height = args.img_height
    img_width = args.img_width
    data_dir = args.data_dir
    load_model = args.load_model
    data_format = args.data_format

    stack_sizes = (n_channels, 48, 96, 192)
    R_stack_sizes = stack_sizes
    A_filter_sizes = (3, 3, 3)
    Ahat_filter_sizes = (3, 3, 3, 3)
    R_filter_sizes = (3, 3, 3, 3)

    # Load previous model if path is given
    if load_model:
        prednet = PredNet(
            stack_sizes,
            R_stack_sizes,
            A_filter_sizes,
            Ahat_filter_sizes,
            R_filter_sizes,
            output_mode="prediction",
            data_format=data_format,
        )
        prednet.load_state_dict(torch.load(load_model))
        prednet.train()
        print("Existing model successsfully lodaded.")
        prednet_dni = PredNetDNI(prednet)
    else:
        raise RuntimeError("Pre-trained prednet model required.")
    print(prednet_dni)
    prednet_dni.cuda()

    assert args.mode == "train"
    train(prednet_dni, args)
    save_path = os.path.join(args.model_dir, "model.pth")
    torch.save(prednet_dni.cpu().state_dict(), save_path)
    if args.evaluate:
        prednet_dni.cuda()
        prednet_dni.eval()
        evaluate(prednet_dni, args)
