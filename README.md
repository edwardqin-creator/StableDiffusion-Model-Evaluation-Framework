# StableDiffusion-Model-Evaluation-Framework
This is a framework to evaluate the Inference Stage of your stable diffusion model

### Support
✅ Evalute the Inference time and GPU memory cost.
<br>✅ Evalute the detailed performance based on Pytorch Profiler and you can use Tensorboard to visualize it.
<br>✅ Evalute the Text2Img quality based on CLIP Score and FID metrics.

### How to use
Take a look at `modules/performance_evaluation.py` and run it

### Quick start
```bash
cd modules
python demo.py
```

### Use tensorboard to virtualize the results
```bash
cd modules
tensorboard --logdir=./log
```

