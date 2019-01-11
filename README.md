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
$ ln -s /exports/eddie/scratch/<UUN>/xbbtagger/preprocessing/input  input
$ ln -s /exports/eddie/scratch/<UUN>/xbbtagger/preprocessing/output output
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


## Jupyter notebook on Eddie

1. Launch a qlogin session, e.g.
   ```bash
   $ qlogin -pe gpu 2
   ```
   or 
   ```bash
   source login.sh gpu 20g
   ```
   
2. Once on the qlogin node, run 
   ```bash
   $ ssh -NR localhost:8882:localhost:8888 login04 &
   ```
   
3. Run 
   ```bash
   $ jupyter notebook --no-browser
   ```
   This will give you an output of text that will look like:
   ```
   ...
   [C 11:59:59.468 NotebookApp]
   
      Copy/paste this URL into your browser when you connect for the first time,
      to login with a token:
          http://localhost:8888/?token=e325bef2289fc2ce991f61a28e36b66c38314f596af30a5f
   ```
   You can ignore all of it apart from the last line (the URL), which you will need to paste into a web browser at a later step.
   
4. Leave that terminal open and in a different terminal window on your local computer run:
   ```bash
   $ ssh -NL localhost:8888:localhost:8882 <UUN>@login04-ext.ecdf.ed.ac.uk
   ```
   This will prompt you for your eddie password - enter it there.

5. Open a web browser and paste the URL that was generated in step 3 into your browser. This will start a jupyter notebook in your browser which will contain the contents of the directory you launched it from on eddie.
