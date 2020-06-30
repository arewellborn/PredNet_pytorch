# -*- coding: utf-8 -*-

from prednet import PredNet
import os
import torch


def model_fn(
    model_dir,
    stack_sizes,
    R_stack_sizes,
    A_filter_sizes,
    Ahat_filter_sizes,
    R_filter_sizes,
    data_format,
):
    """load model function for SageMaker. Must pass in model 
    parameters as kwargs when calling deploy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PredNet(
        stack_sizes,
        R_stack_sizes,
        A_filter_sizes,
        Ahat_filter_sizes,
        R_filter_sizes,
        output_mode="prediction",
        data_format=data_format,
    )
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


# def input_fn(request_body, request_content_type):
#     pass


# def predict_fn(input_object, model):
#     pass


# def output_fn(prediction, response_content_type):
#     pass
