Run something like:
```
ipython train.py -- config/train_shakespeare_char.py --model_type=modify
```
Note that torch.compile doesn't really seem to work on the 2080's
