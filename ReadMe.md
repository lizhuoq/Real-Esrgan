# Real-ERSGAN  
PyTorch Implementation of the Model and Training Process of Real-ESRGAN  
## Dataset  
### Train Dataset  
- DIV2K  
- Flickr2K  
- OutdoorSceneTraining (OST)  
I'll organize these three datasets, and after removing images with height or width less than 256, I'll compress them into a file called "train_HR.zip."  
## Pytorch Weight file  
A weight file for a generator and discriminator model trained for 200 epochs can be download from [here](https://drive.google.com/file/d/1d334zXOshzRkxR4w0E_RL9kNIvStszeB/view?usp=sharing) and [here](https://drive.google.com/file/d/1ttZ2xUNfUO06duYYwWwoyZMONJr9RTQG/view?usp=sharing)(Google Drive Link).  
## Usage  
### Train  
```
python train.py
optional arguments:
  -h, --help            show this help message and exit
  --crop_size CROP_SIZE
                        training images crop size
  --upscale_factor UPSCALE_FACTOR
                        super resolution upscale factor
  --batch_size BATCH_SIZE
                        batch size of train dataset
  --warmup_batches WARMUP_BATCHES
                        number of batches with pixel-wise loss only
  --n_batches N_BATCHES
                        number of batches of training
  --residual_blocks RESIDUAL_BLOCKS
                        number of residual blocks in the generator
  --batch BATCH         batch to start training from
  --lr LR               adam: learning rate
  --sample_interval SAMPLE_INTERVAL
                        interval between saving image samples
```  
### Train details  
- patch size : 256  
- batch size : 16  
- optimizer : Adam
- First train generator for 10k iteration with learning rate $2\times 10^{-4}$  
- Then train generator and discriminator for 4k iteration with learning rate: 1e-4  
- adopt exponential moving average(EMA) with beta = 0.999
- discriminator use Unet with SN(spectral normalization)  
- Real Esrgan is trained with a combination loss of pixel-loss(L1 loss), perceptual loss and GAN loss  
- use {conv1, ..., conv5} feature maps with weights {0.1, 0.1, 1, 1, 1} before activation in the pretrained VGG19 network as perceptual loss  
### Test Single Image  
```
python test_image.py  

optional arguments:
  -h, --help            show this help message and exit
  --upscale_factor UPSCALE_FACTOR
                        super resolution upscale factor
  --test_mode {CPU,GPU}
                        using CPU or GPU
  --image_name IMAGE_NAME
                        test low resolution image name
  --model_name MODEL_NAME
                        generator model epoch name
```  
## Test Single Image Results  
**Upscale Factor = 4**  
low quality image:  
![](assets/Set5_003.png)  
REAL-ESRGAN results:  
![](assets/out_srf_4_Set5_003.png)  
low quality image:  
![](assets/Urban100_083.png)  
REAL-ESRGAN results:  
![](assets/out_srf_4_Urban100_083.png)  
low quality image:  
![](assets/Urban100_100.png)  
REAL-ESRGAN results:  
![](assets/out_srf_4_Urban100_100.png)  
## Training results  
**Upscale Factor = 4**  
This is a batch of images saved every 100 batches.The leftmost column is the low-resolution image obtained by interpolation using the BICUBIC method. The middle column is the original high-resolution image, and the rightmost column is the super-resolution image reconstructed using the model.  
![](images/training/14000.png)