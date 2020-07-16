#  Model-Trainer
Implements a wrapper over the process of training PyTorch models to alleviate the need to explicitly write the commands.

This has been adapted from [here](https://github.com/pytorch/vision/tree/master/references/detection) for simpler models than Mask RCNN for instance.

This code is for simpler models which do not have complex loss functions and make use of whatever is available in PyTorch out of the box.

## Getting Started
You can install *model_trainer* using pip.

``` pip install model-trainer```

## Usage
```
from model_trainer import Trainer

trainer = Trainer(args)
trainer.train_model()
```