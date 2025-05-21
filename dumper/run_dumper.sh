# - -d specifies which GPU device. Normally, 0 is the first GPU installed on your machine, 1 is the second GPU (if there is one), and so forth. If not specified, the default is 0.
# - -s specifies the GPU memory address from which the dumping starts. Note that this address is a physical address. If not specified, the default is 0.
# - -b specifies the number of bytes to dump. 
# - -o specifies the file to which the GPU memory is dumped. 
# - -g optionally specify the active GPU instance (GI). The index depends on the order in which the instances were launched (0 for the first, 1 for the second, etc.). Defaults to 0 if only one instance is active.


./dumper -d 1 -s 0 -b 0xf0000 -o xyz