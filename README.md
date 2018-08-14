# ArioMiner v0.1
### Arionum miner for CPU and GPU 

!!! THIS IS WORK IN PROGRESS, BUGS AND ISSUES ARE TO BE EXPECTED. I CAN'T MAKE ANY PROMISE THAT THIS SOFTWARE WILL WORK PERFECTLY ON YOUR COMPUTER, DON'T USE IT IN PRODUCTION UNTIL THIS MESSAGE WILL NOT BE REPLACED WITH SOMETHING MORE CHEERFUL - ETA: 17.08.2018 !!!

## Dev Fee
In order to support development, this miner has 1% dev fee included - 1 minute from 100 minutes it will mine for developer.

## Features
- optimized argon2 hashing library - both in speed and in memory usage; everything not related to arionum mining was stripped down, indexing calculation was replaced with precalculated versions (improvements in the range of 10% - 50% compared to existing miners)
- support for both CPU and GPU mining
- support for autodetecting the best version of the CPU hasher for your machine (SSE2/SSSE3/AVX2/AVX512F)
- [TODO] support for proxy mode, to act as an aggregator for multiple small miners

## Releases
https://github.com/bogdanadnan/ariominer/releases

## Instructions
What you need:
- recent Linux distribution (Ubuntu recommended) or Mac OS X (support for Windows will be added soon)
- OpenCL libraries and headers (for Ubuntu install **ocl-icd-opencl-dev** package, for Mac OS X it should be included in XCode SDK) - even if you don't plan to use GPU (will add a switch later on to be configurable)
- Git client
- CMake 3
- GCC & G++ v7

Instructions:
- run the following snippet:
```sh
$ git clone http://github.com/bogdanadnan/ariominer.git
$ cd ariominer
$ mkdir build
$ cd build
$ cmake ..
$ make
```
Additional informations:  
https://forum.arionum.com/viewtopic.php?f=15&t=369

Usage:
- starting in miner mode:
```sh
       ariominer --mode miner --pool <pool / proxy address> --wallet <wallet address> --name <worker name> --cpu-intensity <intensity> --gpu-intensity-cblocks <intensity> --gpu-intensity-gblocks <intensity>  
```
- starting in proxy mode:
```sh
       ariominer --mode proxy --port <proxy port> --pool <pool address> --wallet <wallet address> --name <proxy name>
```

Parameters:  
--help: show this help text  
--verbose: print more informative text during run  
--mode <mode>: start in specific mode - arguments: miner / proxy  
- miner: this instance will mine for arionum
- proxy: this instance will act as a hub for multiple miners; useful to aggregate multiple miners into a single instance reducing the load on the pool

--pool <pool address>: pool/proxy address to connect to (eg. http://aropool.com:80)  
--wallet <wallet address>: wallet address; this is optional if in miner mode and you are connecting to a proxy  
--name <worker identifier>: worker identifier this is optional if in miner mode and you are connecting to a proxy  
--port <proxy port>: proxy specific option, port on which to listen for clients this is optional, defaults to 8088  
--cpu-intensity: miner specific option, mining intensity on CPU; value from 0 (disabled) to 100 (full load); this is optional, defaults to 100 (\*)  
--gpu-intensity-cblocks: miner specific option, mining intensity on GPU; value from 0 (disabled) to 100 (full load); this is optional, defaults to 100 (\*)  
--gpu-intensity-gblocks: miner specific option, mining intensity on GPU; value from 0 (disabled) to 100 (full load); this is optional, defaults to 100 (\*)  
--gpu-filter: miner specific option, filter string for device selection; it will select only devices that have in description the specified string; this is optional, defaults to ""  
--force-cpu-optimization: miner specific option, what type of CPU optimization to use; values: REF, SSE2, SSSE3, AVX2, AVX512F; this is optional, defaults to autodetect, change only if autodetected one crashes  
--update-interval: how often should we update mining settings from pool, in seconds; increasing it will lower the load on pool but will increase rejection rate; this is optional, defaults to 2 sec and can't be set lower than that  
--report-interval: how often should we display mining reports, in seconds; this is optional, defaults to 10 sec  

(\*) Mining intensity depends on the number of CPU/GPU cores and available memory. Full load (100) is dynamically calculated by the application.

