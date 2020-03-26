# sagemaker-localmode
Train a model in Sagemaker local mode

# training with estimator (training unet model for segmentation)

```
pip install -r requirements.txt
python script.py --source-dir=$(pwd)/estimator --dataset-path=/path/to/dataset --output-dir=/path/to/output
```

## dataset directory structure

* dataset-root
    * train_frames
        * img
            * ***.jpg
    * train_masks
        * img
            * ***.jpg