
# How to run the proof-of-concept experiment

```bash
cd experiments/poc/
```

1. Generate the proof-of-concept dataset (2D and 3D)

   ```bash
   python generate_poc_data.py
   ```

2. Train the 2D UNet on 2D dataset
  
   ```bash
   python train_2d.py
   ```

3. Locate where the 2D model checkpoint is saved (in ```./tmp/noise0.5/shape/.../model.dat``` and then copy the path to ```POCVoxelEnv.shape_checkpoint``` in ```poc_config.py```
4. Train the 3D UNet on 3D volumesset, with or without 2D pretraining

   ```bash
   python train_3d.py
   ```

The default 3D model is ACSUNet **p.**. To change ACSConv to Conv3d / Conv2_5d or random initialization, modify ```POCVoxelConfig.conv``` and ```POCVoxelConfig.pretrained``` in ```poc_config.py```.
