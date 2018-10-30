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

Copy over the data
```bash
$ mkdir -p /exports/eddie/scratch/<UUN>/xbbtagger/{input,output}
$ ln -s /exports/eddie/scratch/<UUN>/xbbtagger/input input
$ ln -s /exports/eddie/scratch/<UUN>/xbbtagger/output output
$ scp -r
<USERNAME>@lxplus.cern.ch:/afs/cern.ch/work/a/asogaard/public/xbbtagger/input/* /exports/eddie/scratch/<UUN>/xbbtagger/input/
```

Setup the environment
```bash
$ # Install conda...
$ conda env create -f envs/xbbtagger.yml
$ source activate xbbtagger
```

Run the code
```bash
$ python preprocessing.py -m 1 | tee log_preprocessing.out
$ python reweighting.py   -m 1 | tee log_reweighting.out
$ python preparing.py     -m 1 | tee log_preparing.out
$ ls -lrt output/
```