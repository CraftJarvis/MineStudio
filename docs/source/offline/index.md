# Offline Training

Pre-training is a crucial approach for equipping policy models with diverse behaviors, as demonstrated in [VPT](https://arxiv.org/abs/2206.11795). MineStudio supports pre-training with offline data, enabling users to easily perform pre-training through a straightforward configuration file. 

```{note}
The MineStudio offline module is built on [PyTorch Lightning](https://lightning.ai/), providing high flexibility and enabling users to customize it to suit their specific needs.
```

## Quick Start

```{toctree}
:caption: Offline Training with MineStudio

tutorial-vpt
tutorial-rocket
```

## Basic Arguments

`minestudio.offline.trainer.MineLightning` is the core class for offline training. It is a subclass of `lightning.LightningModule` and provides a simple interface for users to customize their training process. 

| Arguments | Description |
| --- | --- |
| `mine_policy` | The policy model to be trained. |
| `callbacks` | A list of objective callbacks to be used during training. |
| `hyperparameters` | A dictionary of hyperparameters to be logged to `wandb`. |
| `log_freq` | The frequency at which logs are uploaded to `wandb`. |
| `learning_rate` | The learning rate for the optimizer. |
| `weight_decay` | The weight decay for the optimizer. |
| `warmup_steps` | The number of warm-up steps for the learning rate scheduler. It is important to train transformer-like networks. |

```{note}
We use `AdamW` as the default optimizer with a linear learning rate scheduler for warmup stage. 
```

```{admonition} Long-Trajectory Training
Due to our advanced data structure, **the offline trainer seamlessly supports long-trajectory training**. By setting `episode_continuous_batch=True` when creating the data module and implementing a memory-based policy, such as a TransformerXL-based policy, the trainer will automatically manage memory iteration for you. 
```


## Objective Callbacks

The loss function is a key component that users often wish to customize when developing new algorithms. MineStudio standardizes this interface and offers a selection of built-in loss functions that users can utilize directly. 

The objective callback template is simple:
```python
class ObjectiveCallback:

    def __init__(self):
        ...

    def __call__(
        self, 
        batch: Dict[str, Any], 
        batch_idx: int, 
        step_name: str, 
        latents: Dict[str, torch.Tensor], 
        mine_policy: MinePolicy
    ) -> Dict[str, torch.Tensor]:
        return {
            'loss': ..., 
            'other_key': ...,
        }
```

```{hint}
`latents` will be returned by the `MinePolicy` object, so users can pass any objective-related information to the callback via `latents` variable. 
```

```{warning}
`loss` term will be added to the final loss function, and all other keys will only be logged to the `wandb` or other loggers. 
```

Here are some examples of built-in objective callbacks:

`````{dropdown} Behavior Cloning Callback
:icon: unlock

The built-in `minestudio.offline.mine_callbacks.BehaviorCloneCallback` looks like this:
```python
class BehaviorCloneCallback(ObjectiveCallback):

    def __init__(self, weight: float=1.0):
        super().__init__()
        self.weight = weight

    def __call__(
        self, 
        batch: Dict[str, Any], 
        batch_idx: int, 
        step_name: str, 
        latents: Dict[str, torch.Tensor], 
        mine_policy: MinePolicy, 
    ) -> Dict[str, torch.Tensor]:
        assert 'agent_action' in batch, "key `agent_action` is required for behavior cloning."
        agent_action = batch['agent_action']
        pi_logits = latents['pi_logits']
        log_prob = mine_policy.pi_head.logprob(agent_action, pi_logits, return_dict=True)
        entropy  = mine_policy.pi_head.entropy(pi_logits, return_dict=True)
        camera_mask = (agent_action['camera'] != 60).float().squeeze(-1)
        global_mask = batch.get('mask', torch.ones_like(camera_mask))
        logp_camera = (log_prob['camera'] * global_mask * camera_mask).sum(-1)
        logp_buttons = (log_prob['buttons'] * global_mask).sum(-1)
        entropy_camera  = (entropy['camera'] * global_mask * camera_mask).sum(-1)
        entropy_buttons = (entropy['buttons'] * global_mask).sum(-1)
        camera_loss, button_loss = -logp_camera, -logp_buttons
        bc_loss = camera_loss + button_loss
        entropy = entropy_camera + entropy_buttons
        result = {
            'loss': bc_loss.mean() * self.weight,
            'camera_loss': camera_loss.mean(),
            'button_loss': button_loss.mean(),
            'entropy': entropy.mean(),
            'bc_weight': self.weight,
        }
        return result
```

While subclassing the `MinePolicy`, one need to return latents (`pi_logits`) in the forward function: 
```python
def forward(self, input, state_in, **kwargs):
    B, T = input["image"].shape[:2]
    first = torch.tensor([[False]], device=self.device).repeat(B, T)
    state_in = self.initial_state(B) if state_in is None else state_in
    (pi_h, v_h), state_out = self.net(input, state_in, context={"first": first})
    pi_logits = self.pi_head(pi_h)
    vpred = self.value_head(v_h)
    latents = {'pi_logits': pi_logits, 'vpred': vpred}
    return latents, state_out
```

`````

`````{dropdown} Kullback–Leibler Divergence Callback
:icon: unlock

The built-in `minestudio.offline.mine_callbacks.KLDivergenceCallback` looks like this:

```python
class KLDivergenceCallback(ObjectiveCallback):
        
    def __init__(self, weight: float=1.0):
        super().__init__()
        self.weight = weight

    def __call__(
        self, 
        batch: Dict[str, Any], 
        batch_idx: int, 
        step_name: str, 
        latents: Dict[str, torch.Tensor], 
        mine_policy: MinePolicy, 
    ) -> Dict[str, torch.Tensor]:
        posterior_dist = latents['posterior_dist']
        prior_dist = latents['prior_dist']
        q_mu, q_log_var = posterior_dist['mu'], posterior_dist['log_var']
        p_mu, p_log_var = prior_dist['mu'], prior_dist['log_var']
        kl_div = self.kl_divergence(q_mu, q_log_var, p_mu, p_log_var)
        result = {
            'loss': kl_div.mean() * self.weight,
            'kl_div': kl_div.mean(),
            'kl_weight': self.weight,
        }
        return result

    def kl_divergence(self, q_mu, q_log_var, p_mu, p_log_var):
        # shape: (B, D)
        KL = -0.5 * torch.sum(
            1 + (q_log_var - p_log_var) - (q_log_var - p_log_var).exp() - (q_mu - p_mu).pow(2) / p_log_var.exp(), dim=-1
        ) # shape: (B)
        return KL
```

While subclassing the `MinePolicy`, one need to return latents (`posterior_dist` and `prior_dist`) in the forward function: (taking GROOT' forward function as an example)
```python
def forward(self, input: Dict, memory: Optional[List[torch.Tensor]] = None) -> Dict:
    ...
    posterior_dist = self.video_encoder(reference)
    prior_dist = self.image_encoder(reference[:, 0])
    ...
    x, memory = self.decoder(x, memory)
    pi_h = v_h = x
    pi_logits = self.pi_head(pi_h)
    vpred = self.value_head(v_h)
    latents = {
        "pi_logits": pi_logits, 
        "vpred": vpred, 
        "posterior_dist": posterior_dist, 
        "prior_dist": prior_dist
    }
    return latents, memory
```
`````
