#!/bin/bash

# Parse out model details
modelId=$MODEL
if [ -z $modelId ]; then
    modelId='category_fnn_model.pkl:1'
fi

categoryRegex="category_([a-z0-9]+)_model"

if [[ $modelId =~ $categoryRegex ]]; then
    categorySvc=$(tr '_' '-' <<< ${BASH_REMATCH[1]})
    serviceName="wbaservice-qa-$categorySvc"
fi

# Test service
data='{"data": [[48522366, 42.0, "N", "N", "N", "N", "N", "N", "N", "N", "N", "N", "N", 0, 0, 0.0, 1, 3, 15.79, 2, 2, 7.44, 1, 3, 16.37, 1, 2, 9.21, 0, 0, 0.0, 8, 8, 750.38, 5, 6, 11.56, 1, 2, 49.55, 1, 1, 19.94, "ENG", "N", "M", "N", 53.79252613207973, 1.550589503639435, "Y", "Y", "N", 13]]}'

az ml service run \
    -g $RESOURCEGROUP \
    -w $WORKSPACENAME \
    -n $serviceName \
    -d "$data"
