# ArioMiner v0.1
### Arionum miner for CPU and GPU 

!!! THIS IS WORK IN PROGRESS, BUGS AND ISSUES ARE TO BE EXPECTED. I CAN"T MAKE ANY PROMISE THAT THIS SOFTWARE WILL WORK PERFECTLY ON YOUR COMPUTER, DON"T USE IT IN PRODUCTION UNTIL THIS MESSAGE WILL NOT BE REPLACED WITH SOMETHING MORE CHEERFUL - ETA: 17.08.2018 !!!

## Features
- optimized argon2 hashing library - both in speed and in memory usage; everything not related to arionum mining was stripped down, indexing calculation was replaced with precalculated versions (improvements in the range of 10% - 50% compared to existing miners)
- support for both CPU and GPU mining (GPU mining is temporarily broken due to the changes for 80k fork - will be fixed shortly)
- support for autodetecting the best version of the CPU hasher for your machine (SSE2/SSSE3/AVX2/AVX512F)
- [TODO] support for proxy mode, to act as an aggregator for multiple small miners

## Instructions
What you need:
- recent Linux distribution (Ubuntu recommended) or Mac OS X (support for Windows will be added soon)
- OpenCL libraries and headers - even if you don't plan to use GPU (will add a switch later on to be configurable)
- Git client
- CMake 3
- GCC v7

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
Usage:
- starting in miner mode:
```sh
       ariominer --mode miner --pool <pool / proxy address> --wallet <wallet address> --name <worker name> --cpu-intensity <intensity> --gpu-intensity <intensity>   
```
- starting in proxy mode:
```sh
       ariominer --mode proxy --port <proxy port> --pool <pool address> --wallet <wallet address> --name <proxy name>
```

Parameters:
 ---help: show this help text
 ---verbose: print more informative text during run
 ---mode <mode>: start in specific mode - arguments: miner / proxy
- miner: this instance will mine for arionum
- proxy: this instance will act as a hub for multiple miners; useful to aggregate multiple miners into a single instance reducing the load on the pool

---pool <pool address>: pool/proxy address to connect to (eg. http://aropool.com:80)
---wallet <wallet address>: wallet address; this is optional if in miner mode and you are connecting to a proxy
---name <worker identifier>: worker identifier this is optional if in miner mode and you are connecting to a proxy
---port <proxy port>: proxy specific option, port on which to listen for clients this is optional, defaults to 8088
---cpu-intensity: miner specific option, mining intensity on CPU; value from 0 (disabled) to 100 (full load); this is optional, defaults to 100 (*)
---gpu-intensity: miner specific option, mining intensity on GPU; value from 0 (disabled) to 100 (full load); this is optional, defaults to 80 (*)
---update-interval: how often should we update mining settings from pool, in seconds; increasing it will lower the load on pool but will increase rejection rate; this is optional, defaults to 2 sec and can't be set lower than that
---report-interval: how often should we display mining reports, in seconds; this is optional, defaults to 10 sec

(*) Mining intensity depends on the number of CPU/GPU cores and available memory. Full load (100) is dynamically calculated by the application.

