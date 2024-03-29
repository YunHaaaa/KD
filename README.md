# Easy to maintain, modular version of 🤗`run_glue.py`

## 👉 See [wandb.ai dashboard](https://wandb.ai/kainoj/run-glue/table?workspace=user-kainoj) here 👈

---

### Evaluate your custom model in 3 steps:
 1. Put code of your model under [`src/models/`](src/models/)
 2. Configure it under [`configs/model/your_model.yaml`](configs/model/)
 3. `python run_glue.py model=your_model` and open your wandb dashboard.

### Evaluate in bulk (no need for bash looping)
```bash
python run_glue.py --multirun \
    experiment=cola,wnli,mrpc \
    model.learning_rate=2e-5,3e-5
```
Will run 3*2 = 6.

### Make GPUs do vroom-voom 🏎
```bash
python run_glue.py --multirun \
    experiment=cola,wnli,mrpc \
    trainer.num_devices=2 \           # 2 gpus per task
    trainer.precision=16 \     # Mixed 16FP precision
    datamodule.batch_size=256  # Beefy batch size
```

### Setup
```bash
conda env create -f environment.yml
conda activate run-glue
```


#### Credits
 - Lightning-hydra template by [ashleve](https://github.com/ashleve/lightning-hydra-template).
 - Some functionalities taken from the original [run_glue.py](https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue.py).
