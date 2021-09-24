# Generation of histopatological images with Pix2pix GAN
This project deals with training a [Pix2pix GAN][1] for generating new histopathological images starting from semantic segmentation mask.
![](result_unitopatho.png)
_From left to right: mask, real image, synthetic image._

## Overview of Code
[dataset](./dataset) directory includes some dataset implemented in Pytorch, used to train the GAN. 
* [pannuke.py](./dataset/pannuke.py) implements consists of the colon images taken from [PanNuke][2] dataset.
You can download the dataset from:
    * [here](https://drive.google.com/uc?id=1_R3jCpMoNBA-vOkd_NJcHamZsv8E3v7Z): the dataset split is 0.9 training and 0.1 test, or
    * [here](https://drive.google.com/uc?id=1cR4FdnoVznh8ZXmAu6AZzbylfYouKRj1): the dataset split is 0.7 training and 0.3 test.
* [unitopatho.py](./dataset/unitopatho.py) is an implementation of [UniToPatho][3], which was taken from [here](https://github.com/EIDOSlab/UNITOPATHO/blob/main/unitopatho.py).
* [unitopatho.py](./dataset/unitopatho_mask.py) inherits from [unitopatho.py](./dataset/unitopatho.py) and adds the feature of the masks to samples of [UniToPatho][3]. 

[train_utils.py](train_utils.py) contains some general-purpose training methods used to train the GAN. For example it contains the method
to train the generator and the discriminator for an epoch, or the methods to do wandb stuffs.
In order to train a GAN, you can launch the following scripts:
* [train_pannuke.py](train_pannuke.py): train the GAN on [PanNuke][2].
* [train_pannuke_ddp.py](train_pannuke_ddp.py): train the GAN on [PanNuke][2] with multi-GPU training.
* [train_utp.py](train_utp.py): train the GAN on [UniToPatho][3].
* [train_utp_ddp.py](train_utp_ddp.py): train the GAN on [UniToPatho][3] with multi-GPU training.

[config.py](config.py) contains the hyperparameters relative to the training and some other parameters.

[utils.py](utils.py) contains some general-purpose methods.

[generator_model.py](generator_model.py) and [discriminator_model.py](discriminator_model.py) are the implementations for the generator and
discriminator architectures used for our trainings

[test.py](test.py) is a file which you can launch for generating the synthetic images of the test set of [UniToPatho][3] 
with a specific model which can be loaded from wandb.


#### Special thanks
Special thanks goes to Aladdin Persson and his [Github repo](https://github.com/aladdinpersson), from which we took some
of his code.

[1]: https://phillipi.github.io/pix2pix/
[2]: https://jgamper.github.io/PanNukeDataset/
[3]: https://ieeexplore.ieee.org/document/9506198