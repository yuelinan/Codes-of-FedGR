Official implementation of ICML 2024 paper "Federated Self-Explaining GNNs with Anti-shortcut Augmentations".

## Data download
- Spurious-Motif: this dataset can be generated via `spmotif_gen/spmotif.ipynb` in [DIR](https://github.com/Wuyxin/DIR-GNN/tree/main). 
- Open Graph Benchmark (OGB): this dataset can be downloaded when running fedgr.sh.


## How to run FedGR?

To train FedGR on OGB dataset:

```python
sh fedgr.sh
```

To train FedGR on Spurious-Motif dataset:

```python
# cd spmotif_codes
sh fedgr_sp.sh
```



