# xbbtagger

To use the necessary python packages on lxplus, _use conda!_


## Preprocessing

Log onto a suitable Eddie node
```bash
$ ssh -Y <UUN>@eddie3.ecdf.ed.ac.uk
$ qlogin -pe sharedmem 4 -l h_vmem=10G
```

Clone the xbbtagger code
```bash
$ cd <YOUR-WORKING-DIRECTORY>
$ git clone git@github.com:asogaard/xbbtagger.git
$ cd xbbtagger
```
In the future, you can use the provided script to log onto Eddie:
```bash
$ source login.sh --help
$ source login.sh gpu 20gb
```

Setup the environment
```bash
$ # Install conda...
$ conda env create -f Environments/xbbtagger.yml
$ source activate xbbtagger
```

Go into the Preprocessing folder
```bash
$ cd Preprocessing
```

Copy over the data
```bash
$ mkdir -p /exports/eddie/scratch/<UUN>/xbbtagger/preprocessing/{input,output}
$ ln -s /exports/eddie/scratch/<UUN>/xbbtagger/preprocessing/ input
$ ln -s /exports/eddie/scratch/<UUN>/xbbtagger/preprocessing/ output
$ scp -r <USERNAME>@lxplus.cern.ch:/afs/cern.ch/work/a/asogaard/public/xbbtagger/input/* /exports/eddie/scratch/<UUN>/xbbtagger/preprocessing/input/
```

Run the preprocessing code
```bash
$ python preprocessing.py --masscut --ttbar | tee log_preprocessing.out
$ python reweighting.py             --ttbar | tee log_reweighting_0.out
$ python reweighting.py   --pt-flat --ttbar | tee log_reweighting_1.out
$ python preparing.py               --ttbar | tee log_preparing.out
$ ls -lrt output/
```


## Training

Log onto a suitable Eddie node
```bash
$ ssh -Y <UUN>@eddie3.ecdf.ed.ac.uk
$ qlogin -pe gpu 2 -l h_vmem=40G
```

Setup the environment
```bash
$ conda env create -f Environments/xbbtagger-gpu.yml
$ module load cuda
$ source activate xbbtagger-gpu
```

Go into the Training folder
```bash
$ cd ../Training
```

Run the training code using TensorFlow (GPU should automatically be inferred if available)
```bash
$ KERAS_BACKEND=tensorflow python btagging_nn.py --input_file ../Preprocessing/output/prepared_sample_v2.h5 --batch_size=8192
```
or using Theano on GPU
```bash
MKL_THREADING_LAYER=GNU THEANO_FLAGS=device=cuda,floatX=float32 python btagging_nn.py --input ../Preprocessing/output/prepared_sample_v2.h5 --batch_size=8192
```

If you want to train multiple classifier on individual pT-slices, please read [_Parameterized Machine Learning for High-Energy Physics_](https://arxiv.org/abs/1601.07913) by Baldi et al. (2016).
If you __still__ want to do it, you can run e.g.
```bash
$ KERAS_BACKEND=tensorflow python btagging_nn.py --pt-slice 200 300
```
which saves the trained model in a unique directory (`KerasFiles/*__pT_200_300GeV/`) which allows you to easily distinguish different models.

To run everything, from reweighting to training, in one go, you can use the provided script
```bash
$ source run.sh
```
