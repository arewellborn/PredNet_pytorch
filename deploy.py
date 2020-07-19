# -*- coding: utf-8 -*-

import os
import argparse
from io import StringIO
import json
import torch
from prednet import PredNet
from evaluate import evaluate


def model_fn(model_dir):
    """load model function for SageMaker. Must pass in model 
    parameters as kwargs when calling estimator.deploy(**kwargs)."""
    n_channels = 3
    stack_sizes = (n_channels, 48, 96, 192)
    R_stack_sizes = stack_sizes
    A_filter_sizes = (3, 3, 3)
    Ahat_filter_sizes = (3, 3, 3, 3)
    R_filter_sizes = (3, 3, 3, 3)
    model = PredNet(
        stack_sizes,
        R_stack_sizes,
        A_filter_sizes,
        Ahat_filter_sizes,
        R_filter_sizes,
        output_mode="prediction",
        data_format='channels_first',
    )
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model


def input_fn(request_body, request_content_type):
    """Pass in args for evaluate function. Takes arguments passed as json.dump(dict)."""
    if request_content_type == "application/json":
        argparse_dict = json.loads(StringIO(request_body))
        parser = argparse.ArgumentParser()
        t_args = argparse.Namespace()
        t_args.__dict__.update(argparse_dict)
        args = parser.parse_args(namespace=t_args)
        return args
    else:
        raise ValueError("Request must be json dump.")


def predict_fn(input_data, model):
    args = input_data
    evaluate(model, args)


# def output_fn(prediction, response_content_type):
#     pass
