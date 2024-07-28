#!/bin/bash
#!/bin/bash

for i in {0..72}; do
    python shuffle_h5.py -CURRENT ${i}
done

for i in {0..72}; do
    python sample_data.py -CURRENT ${i} -BUFFER_SIZE 13888
done

for i in {0..72}; do
    python mixed.py -CURRENT ${i}
done