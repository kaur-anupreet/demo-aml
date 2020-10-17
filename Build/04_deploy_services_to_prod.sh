#!/bin/bash

# Parse out model details
modelId=$MODEL
if [ -z $modelId ]; then
    modelId='category_fnn_model.pkl:1'
fi

categoryRegex="category_([a-z0-9_]+)_model"

if [[ $modelId =~ $categoryRegex ]]; then
    categorySvc=$(tr '_' '-' <<< ${BASH_REMATCH[1]})
    serviceName="wbaservice-prod-$categorySvc"
    modelName="category_${BASH_REMATCH[1]}_model.pkl"
    category=$(tr '[a-z]' '[A-Z]' <<< ${BASH_REMATCH[1]})
fi

# Update scoring file with model name and remove Y value from column list
sed -i "s/<<modelid>>/$modelName/g" './score.py'
sed -i "s/'BOUGHT_CATEGORY_${category}',//g" './score.py'

# Deploy model as a container
az ml model deploy \
    -g $RESOURCEGROUP \
    -w $WORKSPACENAME \
    -n $serviceName \
    -m $modelId \
    --ct wba-prod \
    --ic ../../build/04_inferencingconfig.json \
    --dc ../../build/04_deploymentconfig.json \
    --overwrite