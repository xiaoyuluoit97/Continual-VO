# Dataset preparation

### Replay
You can use the following sh script to prepare the reply dataset.
```
./shuffle.sh
```
### Shuffle dataset
To shuffle all trajectoies of a scence, please run the following .py file, N is the number of current scence
```
python shuffle_h5.py -CURRENT N
```
### Replay sample buffer
Because the Avalanche framework is not compatible with our dataset, we adopted the method of generating the corresponding dataset ourselves to achieve replay.
The size of i can equal to 13888, 5120, 1024. Which represents full size buffer, half size buffer, and 1/10 size buffer data points numbers.
please run the following .py file to sample the buffer
```
python sample_data.py -CURRENT N -BUFFER_SIZE i
```
### Mixed
To mix the buffer and training set
```
python mixed.py -CURRENT N
```
