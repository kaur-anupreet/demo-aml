#!/bin/bash

start=1
stop=10

if [ ! -z "$1" ]
then
    start=stop=$1
fi

images=''
for i in $(seq $start $stop)
do
    # Fetch the latest model
    modelName="category_${i}_model.pkl"
    modelId=$(az ml model show -g $RESOURCEGROUP -w $WORKSPACENAME -n $modelName --query 'id' --output 'tsv')

    # Specify model details
    echo "{\"modelId\":\"$modelId\",\"workspaceName\":\"$WORKSPACENAME\",\"resourceGroupName\":\"$RESOURCEGROUP\"}" > "model_$i.json"
    
    # Update scoring file with model name and remove Y value from column list
    sed "s/<<modelid>>/$modelName/g" './score.py' > "score_$i.py"
    sed -i "s/'BOUGHT_CATEGORY_$i',//g" "./score_$i.py"

    # Build image with retry
    imageName="wbamodelc$i"
    for j in {1..5}
    do
        az ml image create container \
            -g $RESOURCEGROUP \
            -w $WORKSPACENAME \
            -n $imageName \
            -s "./score_$i.py" \
            -r 'python' \
            -f "./model_$i.json" \
            -c './conda.yml' \
            --docker-file './Dockerfile'

        if [ $? -eq 0 ]; then
            # Keep track
            [ ! -z "$images" ] && images+=','
            images+="\"$imageName\""

            break
        fi
    done    
done

# Output list of new images
echo "{\"images\": [$images]}" > "$BUILD_ARTIFACTSTAGINGDIRECTORY/images.json"