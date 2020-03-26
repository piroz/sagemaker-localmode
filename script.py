import os
import argparse

from sagemaker.local import LocalSession
from sagemaker.tensorflow import TensorFlow

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=str, help='path to train script dir')
    parser.add_argument('--dataset-path', type=str, help='path to dataset dir')
    parser.add_argument('--output-dir', type=str, help='path to output dir')
    args = parser.parse_args()

    sagemaker_session = LocalSession()
    sagemaker_session.config = {'local': {'local_code': True}}

    estimator = TensorFlow(entry_point='unet.py',
                        source_dir=args.source_dir,
                        role='/',
                        framework_version='2.0.0',
                        py_version='py3',
                        train_instance_count=1,
                        train_instance_type='local',
                        output_path='file://{}'.format(args.output_dir),
                        sagemaker_session=sagemaker_session)

    print('# Fit model on training data')
    estimator.fit({'train': 'file://{}'.format(args.dataset_path)})

