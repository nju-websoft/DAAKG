# DAAKG
Deep Active Alignment of Knowledge Graph Entities and Schemata, SIGMOD 2023

## Dataset
Link: [DWY100K with dangling entities and schema alignment](https://mega.nz/file/kOcCzYgJ#vqz3qaUOUHA8e746j_4pcTJokH9lSQEX_ioydg7uodI)

## Dependencies
* Python 3.9+
* Python libraries
  * PyTorch: `conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia`
  * PyG: `conda install pyg -c pyg`
  * graph-tool: `conda install -c conda-forge graph-tool`
  * Other libraries: `pip install cupy-cuda11x numpy tensorboard igraph pandas`

## Quick Start
Running script:
```
python -m daakg.run --log transe \
    --data_dir "data/daakg/D_W_100K_V1" \
    --save "output" \
    --rate 0.3 \
    --epoch 1000 \
    --check 10 \
    --update 10 \
    --train_batch_size 1024 \
    --share \
    --encoder "" \
    --hiddens "100" \
    --decoder "TransE" \
    --sampling "T" \
    --k "5" \
    --margin "1" \
    --alpha "1" \
    --feat_drop 0.0 \
    --lr 0.01 \
    --train_dist "euclidean" \
    --test_dist "euclidean"
```

## Citation
If you find our work useful, please kindly cite it as follows:

```
@article{DAAKG_SIGMOD2023,
  author    = { Huang, Jiacheng and Sun, Zequn and Chen, Qijin and Xu, Xiaozhou and Ren, Weijun and Hu, Wei },
  title     = { Deep Active Alignment of Knowledge Graph Entities and Schemata },
  journal   = { Proc. ACM Manag. Data },
  year      = 2023,
  pages     = 159,
  volume    = 1,
  number    = 2
}
```
