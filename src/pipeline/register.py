
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core import Run
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        type=str,
        default='',
        help='Variant name you want to give to the model.'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='outputs',
        help='Location of trained model.'
    )

    args, unparsed = parser.parse_known_args()
    print(args.model_name)
    print(args.model_path)
    
    run = Run.get_context()
    ws = run.experiment.workspace
    
    tags = {
        "runId": str(run.id)
    }

    print(json.dumps(tags))

    model = Model.register(ws, model_name = args.model_name, model_path = args.model_path, tags=tags)

    print('Model registered: {} \nModel Description: {} \nModel Version: {}'.format(model.name, model.description, model.version))
