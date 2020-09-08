
# How to run the LIDC-IDRI experiment

```bash
cd experiments/lidc/
```

1. Download and prepare data
   Download LIDC data from [this link](https://drive.google.com/file/d/1gaMYsgTj0rmnPYGsAFpnB20rWbW-NOll/view?usp=sharing).
   Create data directory and move the downloaded data to ``./data``.
   Unzip data.

   ```bash
   mkdir ./data
   mv <downloaded data> ./data
   unzip <downloaded data>
   ```

2. Run training scripts 
    for segmentation, run

    ```bash
    python train_segmentation.py
    ```

    for classification, run

    ```bash
    python train_classification.py
    ```

The default model is ACS ResNet18 **p.**. To change ACSConv to Conv3d / Conv2_5d or random initialization, or to change to other backbones (DenseNet or VGG16), modify ```LIDCSegConfig``` and ```LIDCClassConfig``` in ```lidc_config.py```, as instructed by the comments aside. For ResNet18, the pretraining methods of Conv3d include [i3d](https://arxiv.org/abs/1705.07750), [video](https://github.com/kenshohara/3D-ResNets-PyTorch) and [mednet](https://github.com/Tencent/MedicalNet). Note that to use pretrained models of [video](https://github.com/kenshohara/3D-ResNets-PyTorch) and [mednet](https://github.com/Tencent/MedicalNet), you need to download and unzip checkpoints from these two links ([video](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M), [mednet](https://drive.google.com/file/d/1399AsrYpQDi1vq6ciKRQkfknLsQQyigM/view?usp=sharing)), and then copy the checkpoint file locations to the corresponding variables of ``LIDCEnv`` in ``lidc_config.py``.


