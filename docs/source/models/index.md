# Models

We provided a template for the Minecraft Policy, `MinePolicy`, and based on this template, we created various different baseline models. Currently, MineStudio supports VPT, STEVE-1, GROOT, and ROCKET-1, among others. This page details the `MinePolicy` template and how to create your own policies.

```{toctree}
:caption: MineStudio Models

baseline-vpt
baseline-steve1
baseline-groot
baseline-rocket1
```

## Quick Start
```{include} quick-models.md
```

## Policy Template (`MinePolicy`)

```{warning}
To ensure compatibility with MineStudio's training and inference pipelines, custom policies **must** inherit from `minestudio.models.base_policy.MinePolicy` and implement its abstract methods.
```

The `MinePolicy` class serves as the base for all policies. Key methods and properties are described below:

````{dropdown} __init__(self, hiddim, action_space=None, temperature=1.0, nucleus_prob=None)

The constructor for the policy.

*   `hiddim` (int): The hidden dimension size for the policy's internal layers.
*   `action_space` (gymnasium.spaces.Space, optional): The action space of the environment. If `None`, a default Minecraft action space (camera and buttons) is used.
*   `temperature` (float, optional): Temperature for sampling actions from the policy head. Defaults to `1.0`.
*   `nucleus_prob` (float, optional): Nucleus (top-p) probability for sampling actions. If `None`, standard sampling is used. Defaults to `None`.

It initializes the policy's action head (`self.pi_head`) and value head (`self.value_head`).

```python
# From minestudio.models.base_policy.py
def __init__(self, hiddim, action_space=None, temperature=1.0, nucleus_prob=None) -> None:
    torch.nn.Module.__init__(self)
    if action_space is None:
        action_space = gymnasium.spaces.Dict({
            "camera": gymnasium.spaces.MultiDiscrete([121]), 
            "buttons": gymnasium.spaces.MultiDiscrete([8641]),
        })
    # self.value_head is a ScaledMSEHead
    self.value_head = ScaledMSEHead(hiddim, 1, norm_type="ewma", norm_kwargs=None)
    # self.pi_head uses the provided action_space, hiddim, temperature, and nucleus_prob
    self.pi_head = make_action_head(action_space, hiddim, temperature=temperature, nucleus_prob=nucleus_prob)
```

```{hint}
Users can override `self.pi_head` and `self.value_head` after calling `super().__init__(...)` if custom head implementations are needed.
```

````

````{dropdown} reset_parameters(self)

Resets the parameters of the policy's action head (`pi_head`) and value head (`value_head`). This can be useful for re-initializing a policy.

```python
# From minestudio.models.base_policy.py
def reset_parameters(self):
    """Resets the parameters of the policy and value heads."""
    self.pi_head.reset_parameters()
    self.value_head.reset_parameters()
```

````

````{dropdown} forward(self, input: Dict[str, Any], state_in: Optional[List[torch.Tensor]] = None, **kwargs) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]

This is an **abstract method** and **must be implemented by derived classes**. It defines the main computation of the policy.

*   `input` (Dict[str, Any]): A dictionary of input tensors (e.g., observations from the environment).
*   `state_in` (Optional[List[torch.Tensor]]): A list of input recurrent state tensors. `None` if the episode is starting or the policy is Markovian.
*   `**kwargs`: Additional keyword arguments.

Returns:
*   `latents` (Dict[str, torch.Tensor]): A dictionary containing at least:
    *   `'pi_logits'` (torch.Tensor): The logits for the action distribution.
    *   `'vpred'` (torch.Tensor): The predicted value function.
*   `state_out` (List[torch.Tensor]): A list containing the updated recurrent state tensors. For a Markovian policy (no state), this should be an empty list (`[]`).

```python
# From minestudio.models.base_policy.py
@abstractmethod
def forward(self,
            input: Dict[str, Any],
            state_in: Optional[List[torch.Tensor]] = None,
            **kwargs
) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
    pass
```
````

````{dropdown} initial_state(self, batch_size: Optional[int] = None) -> List[torch.Tensor]

This is an **abstract method** and **must be implemented by derived classes**. It returns the initial recurrent state of the policy.

*   `batch_size` (Optional[int]): The batch size for which to create the initial state.

Returns:
*   (List[torch.Tensor]): A list of initial state tensors. For a Markovian policy (no state), this should be an empty list (`[]`).

```python
# From minestudio.models.base_policy.py
@abstractmethod
def initial_state(self, batch_size: Optional[int] = None) -> List[torch.Tensor]:
    pass
```
````

````{dropdown} get_action(self, input: Dict[str, Any], state_in: Optional[List[torch.Tensor]], deterministic: bool = False, input_shape: str = "BT*", **kwargs) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]

This method computes and returns an action from the policy based on the current input and state. It's typically used during inference.

*   `input` (Dict[str, Any]): Current observation from the environment.
*   `state_in` (Optional[List[torch.Tensor]]): Current recurrent state.
*   `deterministic` (bool): If `True`, samples actions deterministically (e.g., argmax). If `False`, samples stochastically. Defaults to `False`.
*   `input_shape` (str): Specifies the shape of the `input`.
    *   `"*"`: Single instance input (e.g., one observation at a time during inference). The method handles batching/unbatching internally.
    *   `"BT*"`: Batched sequence input (Batch, Time, ...).
    Defaults to `"BT*"`.
*   `**kwargs`: Additional keyword arguments passed to `forward`.

Returns:
*   `action` (Dict[str, torch.Tensor]): The sampled action.
*   `state_out` (List[torch.Tensor]): The updated recurrent state.

```python
# Simplified from minestudio.models.base_policy.py
@torch.inference_mode()
def get_action(self,
                input: Dict[str, Any],
                state_in: Optional[List[torch.Tensor]],
                deterministic: bool = False,
                input_shape: str = "BT*",
                **kwargs,
) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
    if input_shape == "*":
        # Internal batching for single instance input
        input = dict_map(self._batchify, input)
        if state_in is not None:
            state_in = recursive_tensor_op(lambda x: x.unsqueeze(0), state_in)
    elif input_shape != "BT*":
        raise NotImplementedError("Unsupported input_shape")
    
    latents, state_out = self.forward(input, state_in, **kwargs)
    action = self.pi_head.sample(latents['pi_logits'], deterministic=deterministic)
    self.vpred = latents['vpred'] # Cache for potential later use
    
    if input_shape == "*":
        # Internal unbatching for single instance output
        action = dict_map(lambda tensor: tensor[0][0], action)
        state_out = recursive_tensor_op(lambda x: x[0], state_out)
        
    return action, state_out
```

```{note}
Empirically, setting `deterministic=False` (stochastic sampling) can often improve policy performance during evaluation compared to deterministic actions.
The `input_shape="*"` is common for inference when processing one observation at a time.
```
````

````{dropdown} device (property)

A property that returns the `torch.device` (e.g., 'cpu', 'cuda:0') on which the policy's parameters are located.

```python
# From minestudio.models.base_policy.py
@property
def device(self) -> torch.device:
    return next(self.parameters()).device
```
````

```{hint}
The minimal set of methods you **must** implement in your custom policy are `forward` and `initial_state`.
```

## Your First Policy

Here are basic examples of how to create custom policies by inheriting from `MinePolicy`.

Load necessary modules:
```python
import torch
import torch.nn as nn
from einops import rearrange
from typing import Dict, List, Optional, Tuple, Any

from minestudio.models.base_policy import MinePolicy
# Assuming make_action_head and ScaledMSEHead are accessible for custom heads,
# or rely on those initialized in MinePolicy's __init__.
```

### Example 1: Condition-Free (Markovian) Policy

This policy does not depend on any external condition beyond the current observation and has no recurrent state.

```python
class MySimpleMarkovPolicy(MinePolicy):
    def __init__(self, hiddim, action_space=None, image_size=(64, 64), image_channels=3) -> None:
        super().__init__(hiddim, action_space) # Initializes self.pi_head and self.value_head
        
        # Example backbone: a simple MLP
        # Input image is flattened: image_size[0] * image_size[1] * image_channels
        self.feature_dim = image_size[0] * image_size[1] * image_channels
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, hiddim), 
            nn.ReLU(),
            nn.Linear(hiddim, hiddim),
            nn.ReLU()
        )
        # self.pi_head and self.value_head are already defined in the parent class.

    def forward(self, 
                input: Dict[str, Any], 
                state_in: Optional[List[torch.Tensor]] = None, # Will be None for Markovian
                **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        # Assuming input['image'] is (B, T, H, W, C)
        # For a Markovian policy, we typically expect T=1 or process frames independently.
        # If T > 1, this example processes each time step independently.
        img_obs = input['image'] # Shape: (B, T, H, W, C)
        b, t, h, w, c = img_obs.shape
        
        # Flatten image: (B, T, H, W, C) -> (B*T, H*W*C)
        # Normalize image (example: scale to [0,1])
        x = rearrange(img_obs / 255.0, 'b t h w c -> (b t) (h w c)')
        
        features = self.net(x) # Shape: (B*T, hiddim)
        
        # Reshape for policy and value heads if they expect (B, T, hiddim)
        features_reshaped = rearrange(features, '(b t) d -> b t d', b=b, t=t)
        
        pi_logits = self.pi_head(features_reshaped) # pi_head handles (B, T, D) or (B*T, D)
        vpred = self.value_head(features_reshaped)  # value_head handles (B, T, D) or (B*T, D)
        
        result = {
            'pi_logits': pi_logits, 
            'vpred': vpred, 
        }
        # For a Markovian policy, state_out is an empty list
        return result, []

    def initial_state(self, batch_size: Optional[int] = None) -> List[torch.Tensor]:
        # Markovian policy has no state, return empty list
        return []

```

### Example 2: Condition-Based (Markovian) Policy

This policy takes an additional 'condition' tensor as input.

```python
class MySimpleConditionedPolicy(MinePolicy):
    def __init__(self, hiddim, action_space=None, image_size=(64, 64), image_channels=3, condition_dim=64) -> None:
        super().__init__(hiddim, action_space)
        
        self.feature_dim = image_size[0] * image_size[1] * image_channels
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, hiddim), 
            nn.ReLU(),
        )
        # Embedding for the condition
        self.condition_net = nn.Linear(condition_dim, hiddim) # Or nn.Embedding if condition is discrete
        
        # Fusion layer
        self.fusion_net = nn.Sequential(
            nn.Linear(hiddim * 2, hiddim), # Example: concatenate image and condition features
            nn.ReLU(),
            nn.Linear(hiddim, hiddim),
            nn.ReLU()
        )

    def forward(self, 
                input: Dict[str, Any], 
                state_in: Optional[List[torch.Tensor]] = None, 
                **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        img_obs = input['image'] # Shape: (B, T, H, W, C)
        condition = input['condition'] # Shape: (B, T, condition_dim)
        
        b, t, h, w, c = img_obs.shape
        
        x_img = rearrange(img_obs / 255.0, 'b t h w c -> (b t) (h w c)')
        img_features = self.net(x_img) # Shape: (B*T, hiddim)
        
        # Process condition
        # Assuming condition is already (B, T, condition_dim) -> (B*T, condition_dim)
        cond_features = self.condition_net(rearrange(condition, 'b t d -> (b t) d')) # Shape: (B*T, hiddim)
        
        # Fuse features (example: concatenation)
        fused_features = torch.cat([img_features, cond_features], dim=-1)
        final_features = self.fusion_net(fused_features) # Shape: (B*T, hiddim)

        final_features_reshaped = rearrange(final_features, '(b t) d -> b t d', b=b, t=t)
        
        pi_logits = self.pi_head(final_features_reshaped)
        vpred = self.value_head(final_features_reshaped)
        
        result = {
            'pi_logits': pi_logits, 
            'vpred': vpred, 
        }
        return result, []

    def initial_state(self, batch_size: Optional[int] = None) -> List[torch.Tensor]:
        return []
```

```{warning}
These examples are simplified for demonstration. Real-world policies, especially for complex environments like Minecraft, often require more sophisticated architectures (e.g., CNNs for image processing, recurrent layers like LSTMs or Transformers for temporal dependencies if not Markovian).
The `input['image']` format and normalization (e.g., `/ 255.0`) should match how your environment provides observations and how your model expects them.
```


