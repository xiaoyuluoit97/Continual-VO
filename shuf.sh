#!/bin/bash
#!/bin/bash

# 循环从 0 到 25 的范围
for i in {0..13}; do
    # 创建名为 train_fulreplay_${i}.h5 的空文件，大小为 0 字节
    python mixed.py -CURRENT ${i}
done

