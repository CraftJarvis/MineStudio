# Customize

Our framework supports customization of online algorithm details.

## Trainer

To customize the trainer, you need to imply an class inherit from `minestudio.online.trainer.basetrainer.BaseTrainer` with and implement the abstract methods `setup_model_and_optimizer` and `train`. 

`setup_model_and_optimizer` return a pair `(model, optimizer)`.

In the `train` function, you need to define the training loop. You can use the `fetch_fragments_and_estimate_advantages` method to obtain data from the replay buffer.

```python
from minestudio.online.trainer.basetrainer import BaseTrainer
class PPOTrainer(BaseTrainer):
    def setup_model_and_optimizer(self):
        # Define model and optimizer
        pass

    def train(self):
        # Custom training logic
        pass
```

Refer to `minestudio.online.trainer.ppotrainer.PPOTrainer` for an example. 
