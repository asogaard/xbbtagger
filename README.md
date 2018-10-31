# xbbtagger

To use the necessary python packages on lxplus, _use conda!_

### Quick start

Log onto a suitable Eddie node
```bash
$ ssh -Y <UUN>@eddie3.ecdf.ed.ac.uk
$ qlogin -pe sharedmem 4 -l h_vmem=10G
```
Actually, 20 GB (e.g. 2 x 10 GB) in total should be sufficient, cf. the
print-outs in the running code below. We're requesting 40 GB just to be safe.

Clone the xbbtagger code
```bash
$ cd <YOUR-WORKING-DIRECTORY>
$ git clone git@github.com:asogaard/xbbtagger.git
$ cd xbbtagger
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
$ scp -r
<USERNAME>@lxplus.cern.ch:/afs/cern.ch/work/a/asogaard/public/xbbtagger/input/* /exports/eddie/scratch/<UUN>/xbbtagger/preprocessing/input/
```

Run the preprocessing code
```bash
$ python preprocessing.py -m 1 | tee log_preprocessing.out
$ python reweighting.py   -m 1 | tee log_reweighting.out
$ python preparing.py     -m 1 | tee log_preparing.out
$ ls -lrt output/
```

Go into the Training folder
```bash
$ cd ../Training
```

Run the training code
```bash
$ python btagging_nn.py --input_file ../Preprocessing/output/prepared_sample_v2.h5
```
