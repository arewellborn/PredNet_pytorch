# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import tarfile
import json
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
from torch.autograd import Variable

# zcr lib
from prednet import PredNet
from prednet_dni import PredNetDNI
from data_utils import ZcrDataLoader

# Sagemaker imports
import boto3


def arg_parse():
    desc = "Video Frames Predicting Task via PredNet."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--mode", default="evaluate", type=str, help="train or evaluate (default: evaluate)"
    )
    parser.add_argument(
        "--load-dni-model",
        default="",
        type=str,
        help="Path to pre-existing model that can be loaded before training.",
    )
    parser.add_argument(
        "--seed", default=1234, type=int, help="Random seed for training.",
    )
    parser.add_argument(
        "--epochs",
        default=2,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--include-datetime", default=True, type=bool, help="Whether to return datetimes from the dataloader.",
    )
    parser.add_argument(
        "--dni-offset", default=0, type=int, help="Offset DNI output by one step size (default: 0).",
    )
    parser.add_argument(
        "--checkpoint_file",
        default="",
        type=str,
        help="checkpoint file for evaluating. If using remote S3 repository, this must be the key. (default: none)",
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, metavar="N", help="The size of batch"
    )
    parser.add_argument(
        "--num_plot", default=40, type=int, metavar="N", help="how many images to plot"
    )
    parser.add_argument(
        "--num_timeSteps",
        default=10,
        type=int,
        metavar="N",
        help="number of timesteps used for sequences in training (default: 10)",
    )
    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--step",
        default=1,
        type=int,
        help="The step size for image sequences (default: 1)",
    )
    parser.add_argument("--shuffle", default=False, type=bool, help="shuffle or not")
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
        "--bucket",
        default="",
        type=str,
        help="S3 bucket that contains the model artifacts.",
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


def evaluate(model, args):
    """Evaluate PredNet on KITTI sequences"""
    prednet_dni = model  # Now prednet_dni is the testing model (to output predictions)

    DATA_DIR = args.data_dir
    RESULTS_SAVE_DIR = args.output_data_dir
    test_file = os.path.join(DATA_DIR, "validation.h5")
    test_sources = os.path.join(DATA_DIR, "sources_validation.h5")

    output_mode = "prediction"
    sequence_start_mode = "all"
    N_seq = None
    # Set up dataloader
    dataLoader = ZcrDataLoader(
        test_file, test_sources, output_mode, sequence_start_mode, N_seq, args
    ).dataLoader()
    # Set up initial states
    input_shape = (
        args.batch_size,
        args.num_timeSteps,
        args.n_channels,
        args.img_height,
        args.img_width,
    )
    initial_states_dni = prednet_dni.get_initial_states(input_shape)
    states = initial_states_dni
    # Generate predictions
    prediction_target = []
    for step, (frameGroup, target, datetimes) in enumerate(dataLoader):
        batch_frames = Variable(frameGroup.cuda())
        output = prednet_dni(batch_frames, states)
        target = target[:, -1]
        datetimes = datetimes[:, -1]
        additions = list(
            zip(
                datetimes.cpu().detach().numpy().astype(str), 
                output.cpu().detach().numpy(), 
                target.cpu().detach().numpy()
            )
        )
        prediction_target += additions

    # Save predictions and targets in a csv
    save_dir = os.path.join(RESULTS_SAVE_DIR, "predictions")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_loc = os.path.join(save_dir, "prediction_target.csv")
    with open(save_loc, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for row in prediction_target:
            writer.writerow(row)

    print('The DNI data are saved in "%s"! Have a nice day!' % save_dir)


def load_pretrained_weights(model, state_dict_file):
    """直接使用从原作者提供的Keras版本的预训练好的PredNet模型中拿过来的参数"""
    model = model.load_state_dict(torch.load(state_dict_file))
    print("weights loaded!")
    return model


if __name__ == "__main__":
    args = arg_parse()
    print_args(args)

    n_channels = args.n_channels
    img_height = args.img_height
    img_width = args.img_width
    data_format = args.data_format
    load_dni_model = args.load_dni_model

    # stack_sizes       = eval(args.stack_sizes)
    # R_stack_sizes     = eval(args.R_stack_sizes)
    # A_filter_sizes    = eval(args.A_filter_sizes)
    # Ahat_filter_sizes = eval(args.Ahat_filter_sizes)
    # R_filter_sizes    = eval(args.R_filter_sizes)

    stack_sizes = (n_channels, 48, 96, 192)
    R_stack_sizes = stack_sizes
    A_filter_sizes = (3, 3, 3)
    Ahat_filter_sizes = (3, 3, 3, 3)
    R_filter_sizes = (3, 3, 3, 3)

    def load_model_fn(load_model):
        if ".pth" in load_model:
            load_model = load_model
        elif ".tar.gz" in load_model:
            tar = tarfile.open(load_model, "r:gz")
            outpath = load_model.rsplit("/", 1)[0]
            tar.extractall(path=outpath)
            tar.close()
            load_model = os.path.join(outpath, "model.pth")
        else:
            raise RuntimeError("File extension not recognized.")
        return load_model
    
    # Load previous model if path is given
    if load_dni_model:
        prednet = PredNet(
            stack_sizes,
            R_stack_sizes,
            A_filter_sizes,
            Ahat_filter_sizes,
            R_filter_sizes,
            output_mode="prediction",
            data_format=data_format,
        )
        load_model = load_model_fn(load_dni_model)
        prednet_dni = PredNetDNI(prednet)
        prednet_dni.load_state_dict(torch.load(load_model))
        prednet_dni.eval()
        print("Existing PredNetDNI model successsfully lodaded.")
    else:
        raise RuntimeError("Pre-trained PredNet or PredNetDNI model required.")
    print(prednet_dni)
    prednet_dni.cuda()

    ## 直接使用作者提供的预训练参数
    # state_dict_file = './model_data_keras2/preTrained_weights_forPyTorch.pkl'
    # # prednet = load_pretrained_weights(prednet, state_dict_file)   # 这种不work... why?
    # prednet.load_state_dict(torch.load(state_dict_file))

    assert args.mode == "evaluate"
    evaluate(prednet_dni, args)
