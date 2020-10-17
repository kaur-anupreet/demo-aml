#!/bin/bash

az extension add -n azure-cli-ml

pip install -U azureml-sdk[automl]