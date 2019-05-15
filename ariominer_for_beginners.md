# First steps in Arionum mining

Arionum coin mining is different than other cryptocurrencies in a couple of ways:
- has 2 alternating block types, named CBlock (the ones with even height) and GBlock (the ones with odd height). The blocks use different algorithm settings. The names have a historical reason. The CBlocks were meant to be easily mined using CPU devices and GBlocks by GPU devices. Mining them the other way around should have been very inneficient. A lot of progress was made in mining code so at the current stage both CPU and GPU devices can mine both kind of blocks quite efficiently. GPUs are much faster than CPUs though in both blocks but also consumes much more energy. As they use different algorithm settings, you will have 2 alternating hashrate during mining, a lower hashrate for CBlocks and a higher one (aprox 10 times higher) for GBlocks.
- has a proprietary protocol to communicate with the pools (no stratum support, at least not at the moment of writing these docs). This means there is a small number of pools available for mining. The main ones for the moment are http://aropool.com (managed by arionum developers) and http://mine.arionumpool.com (managed by HashPi, you can find him on Discord)
- the wallet and the node are 2 separate applications. The node contains the actual blockchain data, while the wallet connects to the node for retrieving balance and making transactions. You do not need to run a node in order to use the wallet, it will automatically connect to a running node on the network. So, you don't need to download gigabytes of data to use and mine arionum. There are 3 wallets available:
  - lightWalletCLI (https://github.com/arionum/lightWalletCLI) - this is a PHP based, console mode version; it is recommended for linux users accustomed with console usage
  - lightWalletGUI (https://github.com/arionum/lightWalletGUI) - visual basic code, for Windows users
  - ArionumElectron (https://github.com/CuteCubed/Arionum-Electron) - javascript/nodejs based, crossplatform wallet
  
In order to mine arionum using ariominer you need to do the following:
- download one of the wallets and use it to generate an arionum address. If you are using lightWalletCLI, when running it first time it will create an arionum address. The output will be something like:

```
$./light-arionum-cli 
No ARO wallet found. Generating a new wallet!
Would you like to encrypt this wallet? (y/N) N
Your Address is: QKMm3mLeZabAVshn8BLBcJ4BRVi3PbkPp9p7d7GPva6k6vRdu5fMEoZS1ZRHbvaUKx65ha1Vnb1JE5dKb5SV2PM
Your Public Key is: PZ8Tyr4Nx8MHsRAGMpZmZ6TWY63dXWSD1Gms6T7iy3UxSMVtfMXTsjnXvtnWtxpgFT8Cdgd7qgFgMU1zLS5GAioKTfc7zRoybVHdH2yMKEKFsMggQELtCmpc
Your Private Key is: xxxx
```
- download an ariominer binary package from release section (https://github.com/bogdanadnan/ariominer/releases) or build it yourself from source code based on instructions in Readme file
- ariominer has the ability to use any available hardware you have (CPU or GPU or both). In order to control which device it will use and how much of it, ariominer uses the notion of intensity. In case of CPU, the intensity is the percent of cores that the miner will use. For example if you have a quadcore, and use an intensity of 50% for CPU, it will use 2 cores. For GPUs on the other hand, intensity is a much more complex stuff. It represents the percent of the total GPU memory used by the miner. Because each block type algorithm use a different amount of memory for a single hash (~96MB for CBlocks and 16MB for GBlocks), total amount of memory used divided by memory used by a single hash will give you the number of "threads" run on GPU (or individual hashes calculated in a GPU batch). Because of the way GPUs work internally, the hashrate doesn't increase proportional with intensity for GPUs. It actually follows a sawtooth pattern. And the pattern is different for CBlocks and GBlocks. So in order to best use your GPUs, ariominer gives you the possibility to specify different intensity for each. To summarize you will have 3 parameters to optimize in order to get the highest hashrate:
  - --cpu-intensity: this represents the percent of cores that will be used by CPU for both types of blocks. Higher intensity means higher hashrate. If you run the miner with GPUs as well, and you have several cards, you might want to use a low intensity for CPU mining or even disabling it completely (by setting it to 0), because GPUs also need some CPU power to prepare the work for them. 
  - --gpu-intensity-cblocks: this represents the intensity used for CBlocks on GPU. CBlocks use a high amount of memory per hash, so for best hashrate you will want this number to be as high as possible (even 100). Because the GPU memory is also used by the system for video display you might not be able to set it to 100 though, so just try numbers from 100 and decrease by 1 until it doesn't give any error.
  - --gpu-intensity-gblocks: this represents the intensity used for GBlocks on GPU. This is a very tricky one. It follows a sawtooth pattern and finding the best value manually is difficult. The best values are usually somewhere between 30 and 50, but even values different with a single unit might give you huge hashrate boosts (remember sawtooth pattern?). In order to help you find the perfect value, ariominer has a special mode called autotune that will go through all intensity values in an interval specified by you and will measure hashrate, giving you in the end the best value to use. I'll show you how to use it in a few moments.
- in order to use different devices, ariominer implements them as hashers. There are 5 hashers for CPU (depending on instruction support: SSE2, SSSE3, AVX, AVX2, AVX512F) and 3 hashers for GPU (OPENCL - for all card types, CUDA - for NVidia cards, AMDGCN - for some AMD cards, RX and VEGA mainly). Ariominer tries its best to autodetect the hardware you have and to use the best possible hasher for it. But for specific hardware/software configurations, autodetection might not work well. Especially on Windows systems, AVX2 instructions are not properly detected. You can force it to use specific hashers using the following arguments: 
  - --force-cpu-optimization <SSE2, SSSE3, AVX, AVX2, AVX512F> (I really suggest checking if your processor has AVX2 and if so use the flag to force it, AVX2 almost doubles hashrate for CPUs)
  - --force-gpu-optimization <OPENCL, CUDA, AMDGCN>
- having these informations, you can start mining using the follwing command line:
```
./ariominer --mode miner --pool <address of pool / http://aropool.com> --wallet <your address / QKMm3mLeZabAVshn8BLBcJ4BRVi3PbkPp9p7d7GPva6k6vRdu5fMEoZS1ZRHbvaUKx65ha1Vnb1JE5dKb5SV2PM> --name <worker name / miner1> --cpu-intensity <percent of cores used / 80> --gpu-intensity-cblocks <percent of gpu memory used by cblocks / 100> --gpu-intensity-gblocks <percent of gpu memory used by gblocks / 30> 
```
- as I mentioned earlier, GBlocks intensity value for GPUs is difficult to optimize by hand. You can make ariominer to find the best value using the following command:
```
./ariominer --mode autotune --autotune-start 20 --autotune-stop 60 --block-type GPU
```

- this command will check each value between 20 and 60 averaging hashrate over 20 seconds for each step, and will allow you to find the best possible intensity for your system for GBlocks. You can run it for CBlocks as well using --block-type CPU, but as I said, for those is better to use a high value, close to 100 so is easier to optimize manually.

## Specific use cases and questions

### I have several cards and I want to use only some of them. 
In this case use --gpu-filter argument. Please keep in mind that this is actually a filter and accepts a string as a filter. When you start the miner it detects your cards and displays a list of them with an index in front (eg. [1] Intel - Iris Pro (1.5GB)). The filter argument will actually check that text if it has the filter in it. For example, --gpu-filter AMD will match all cards that have AMD in their name. Or --gpu-filter [1] will match all cards that have the text [1] in the name. You can specify multiple filters using comma separator. For example --gpu-filter [1],[2],[3] will use the cards having [1] or [2] or [3] in the name, so basically the first 3 cards in the system. Be careful not to use spaces between filters, just comma.

### I have a mix of cards in the system, all from the same vendor (NVidia or AMD). 
In this case the best intensity for each card is different. Same as for the filter, you can specify multiple intensity values by separating them with comma. They will be matched with the cards based on the order. For example --gpu-intensity-gblocks 35,53,54 means it will use 34 for first cards, 53 for the second and 54 for the third.

### I have a mix of cards in the system from different vendors (both NVidia and AMD). 
This can be enabled by specifying multiple hashers in --force-gpu-optimization flag. For example --force-gpu-optimization CUDA,AMDGCN will use CUDA for NVIDIA cards and AMDGCN for AMD cards. This specific combination (CUDA,AMDGCN) can be used directly without any additional settings because each hasher will use different cards. Using a combination of CUDA with OPENCL or AMDGCN with OPENCL is much more tricky because OPENCL will probably autodetect the cards used by CUDA and AMDGCN as well and the miner will just crash in this case. To help him decide which cards to use for OPENCL you will have to use a special form of the filter flag, example: --force-gpu-optimization CUDA,OPENCL --gpu-filter CUDA:NVidia,OPENCL:AMD. This translates to: for CUDA use only cards having NVidia in the name and for OPENCL use only cards having AMD in the name.
### I have a card that is randomly crashing but the miner doesn't detect this and continues to mine without it until I restart. 
You can use --chs-threshold and --ghs-threshold to specify a value for cblocks and gblocks hashrate under which the miner will automatically exit. Using a loop in bash or bat file you can force it in this way to automatically restart if hashrate goes under your specified value. The hashrate needs to be under that value for at least 5 reports before the exit is triggered (~50 sec). It is built as such in order to allow for the system to stabilize at startup or after block change.
### I want to integrate this with specific mining monitor software, is there an API available? 
Yes there is, you need to use --enable-api-port <value greater than 1024> to enable it. Once you add this argument, you can get status reports at http://localhost:<api_port>/status link. This will return you a JSON with a lot of internal details. Btw, there is a hiveos package already built in case you want to use it, you can find it on release page (https://github.com/bogdanadnan/ariominer/releases).
### I have many small miners, is there any way to aggregate them in a single worker instead of directing them individually to the pool?
That's the proxy mode for. Ariominer has a builtin proxy that can act as a pool relay for many small miners and which will relay the requests to a pool of your choice. There are changes related to dev fee when you run the miners through the proxy. In that case, the dev fee is disabled at the miner side and instead is collected by the proxy. That is needed because the proxy overwrites the wallet/pool settings sent by the miner with its own values, so dev fee can't be collected anymore from miner. You can start ariominer in proxy mode using the following syntax:
```
./ariominer --mode proxy --port <proxy port> --pool <pool address> --wallet <wallet address> --name <proxy name>
```

After you start it, redirect all your miners to point to the address and port of your proxy. The wallet used for the miners is irrelevant as it will be replaced by the wallet set on the proxy. There is a nice dashboard embedded into ariominer that allows you to get a lot of statistics from last 24h. You can check it out by visiting the proxy address in a browser: http://<proxy ip>:<proxy port> . Please keep in mind that proxy support is an experimental feature.  
### Ariominer is trying to connect to coinfee.changeling.biz, what is this?
This site has the dev fee settings to use (dev wallet and pool to connect to during that 1 min period). I implement it as such in order to be able to change the wallet in case the current one becomes compromised or to change the pool to a specific one in the future. Please don't block the site, there is no malicious code run by ariominer. The source code is open and if you don't trust the binaries you can always compile it yourself and check the code. 
### How can I solo mine?
There is no support (yet) for solo mining. I might add it in the future as it is not a difficult task, but for the moment the need for it was not big enough.
  
  

