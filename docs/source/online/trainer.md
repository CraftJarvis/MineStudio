# Trainer

This document describes the trainer in the online training module, with the primary implementation being the `PPOTrainer`.

## Design Principle

The `PPOTrainer` (Proximal Policy Optimization Trainer) is the core component responsible for updating the policy and value function models. It leverages the experience data collected by rollout workers and is architected for efficient, distributed training using `Ray Train`.

The design of the `PPOTrainer` is guided by several key principles. At its heart is the **PPO Algorithm**, a well-regarded actor-critic method in reinforcement learning known for its balance of sample efficiency, stability, and ease of implementation. To handle the demands of large-scale training, the trainer is built for **Distributed Training**, utilizing `Ray Train` for data parallelism. This enables the training process to scale across multiple GPUs and even multiple nodes in a cluster.

Effective **Data Handling** is crucial. The trainer ingests `SampleFragment`s (segments of trajectories) from a `ReplayBuffer` (a process managed by its `BaseTrainer` parent class). It then computes Generalized Advantage Estimation (GAE) and TD-targets, which are vital for PPO's policy and value function updates. A `data_iter` is employed to efficiently load and prepare batches of data for the training epochs.

**Model Management** encompasses several responsibilities. The trainer initializes the `MinePolicy` model and an AdamW optimizer. It can optionally maintain a separate `ref_model` for KL divergence regularization, which helps stabilize training by penalizing large deviations from a previous policy. Robust model checkpointing capabilities are included, allowing for saving and loading of training progress. Furthermore, updated model weights are periodically broadcast to the rollout workers to ensure they are using the latest policy for experience collection.

The **Loss Calculation** in PPO is a composite objective. This typically includes:
*   A **Policy Loss**, based on a clipped surrogate objective to ensure stable policy updates.
*   A **Value Function Loss**, usually a mean squared error to train the value function, which can also be clipped.
*   An optional **Entropy Bonus**, added to the loss to encourage exploration by discouraging the policy from becoming too deterministic too quickly, as stated in Soft Actor-Critic (SAC) literature.
*   An optional **KL Divergence Penalty**, based on the KL divergence between the current policy and the `ref_model`.

To aid convergence, the trainer supports **Learning Rate Scheduling**, often in the form of linear annealing. **Gradient Management** techniques are also incorporated, such as gradient accumulation (to simulate larger batch sizes) and gradient norm clipping (to prevent exploding gradients). Finally, comprehensive **Logging and Monitoring** are achieved using `wandb_logger` for Weights & Biases integration and `torchmetrics` for accumulating various performance statistics.

## Logic

The training process orchestrated by the `PPOTrainer` unfolds in several key stages:

### 1. Initialization

The journey begins with the setup phase, primarily within the `__init__` and `setup_model_and_optimizer` methods. First, all necessary **Hyperparameters** for the PPO algorithm, optimizer settings, batch sizes, and the training schedule are configured.

Following this, the **Model and Optimizer are Instantiated**. This involves creating the primary policy model (an instance of `MinePolicy`) and initializing the AdamW optimizer with the configured learning rate and weight decay.
```{note}
If the `zero_initial_vf` option is enabled, the weights of the value function head within the policy model are explicitly initialized to zero. This technique can sometimes provide a better starting point for the value function in the early stages of training.
```
If KL divergence regularization is active (i.e., `kl_divergence_coef_rho` is non-zero), a separate **Reference Model (`ref_model`) is also set up**. This model typically holds an older version of the policy and serves as a baseline for the KL penalty. Lastly, the model is prepared for **Distributed Setup** with `Ray Train`, which usually involves wrapping it with `torch.nn.parallel.DistributedDataParallel` if multiple GPUs or nodes are part of the training cluster.

### 2. Main Training Loop

The `train` method orchestrates the overarching training loop. This loop **Iterates** for a predefined `num_iterations`. Within each iteration, if **Learning Rate Annealing** (`anneal_lr_linearly`) is enabled, the learning rate for the optimizer is adjusted, typically decreased linearly as training progresses. The core work of data processing and model updates for each cycle happens within the `train_iteration` method.

A critical aspect of distributed training is **Model Broadcasting**. This is typically handled by the rank 0 worker (the chief worker). After an initial `vf_warmup` period (during which the value function might be trained more aggressively or exclusively to stabilize it), the updated model weights are broadcast to all rollout workers. This synchronization is facilitated by the `broadcast_model_to_rollout_workers` method, often part of the `BaseTrainer` class, ensuring that data collection agents are working with the most up-to-date policy.

### 3. Single Training Iteration

Each call to the `train_iteration` method performs one complete pass of training. It begins with **Data Acquisition and Preprocessing**. The `fetch_fragments_and_estimate_advantages` method (inherited or called from `BaseTrainer`) is invoked. This is a crucial step that retrieves a collection of `SampleFragment`s from the replay buffer, computes Generalized Advantage Estimation (GAE) values, and calculates TD-lambda targets for value function training.

Once the data is prepared and advantages are estimated, the **PPO Update Execution** commences by calling the `ppo_update` method with this processed information. Additionally, if a KL divergence penalty is used, its coefficient (`kl_divergence_coef_rho`) is decayed according to `coef_rho_decay`, often to gradually reduce its influence as training stabilizes.

### 4. PPO Update Step

The `ppo_update` method is where the core PPO algorithm refines the model parameters. This process itself has several sub-stages. The update iterates for a specified number of **Epochs** (`epochs_per_iteration`) over the currently collected batch of data. Within each epoch, a `data_iter` provides **Mini-Batches** of `SampleFragment`s.

For each mini-batch, several operations occur during **Mini-Batch Processing**. First, `prepare_batch` converts the list of `SampleFragment`s into a batched PyTorch tensor format, making it suitable for feeding into the neural network. Data is often processed in fixed-length temporal chunks, defined by `context_length`.

The heart of the update is the **Forward Pass**. The current policy model processes the data chunk to obtain new action log probabilities (`new_logp`) for the actions taken in the fragments, value predictions (`vpred`), and the raw policy logits. If KL regularization is active, the `ref_model` also performs a forward pass on the same data to get its corresponding action log probabilities or logits.

Next comes the **Loss Calculation**. The PPO objective is a composite of several terms:
*   **Policy Loss (Actor Loss)**: This is calculated using the PPO clipped surrogate objective. It involves computing the probability ratio $r_t(\theta) = \exp(\log \pi_\theta(a_t|s_t) - \log \pi_{\theta_old}(a_t|s_t))$. The loss is then derived from $\min(r_t(\theta)\hat{A}_t, \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)`, where `\hat{A}_t` is the advantage estimate. This clipping mechanism is key to PPO's stability.
*   **Value Loss (Critic Loss)**: This is typically the Mean Squared Error (MSE) between the predicted values (`vpred`) and the calculated TD-targets. The value loss can also be clipped (if `clip_vloss` is true) by comparing it to a version where value predictions are clipped around the `old_vpred` from the previous iteration.
*   **Entropy Bonus**: An entropy term, derived from the policy logits, is added to the objective. This encourages exploration by penalizing policies that become too deterministic too quickly.
*   **KL Divergence Loss**: If enabled, this measures the KL divergence between the current policy's action distribution and that of the `ref_model`. It acts as an additional regularization term.
The **Total Loss** is then a weighted sum of these components.
```{tip}
During an initial `vf_warmup` phase, the training might focus more on the value function. In such cases, the total loss might be dominated by, or solely consist of, the value function loss (and KL loss if active). This helps in stabilizing the value estimates before the policy is trained more aggressively.
```
With the total loss computed, the **Backward Pass and Optimization** step follows. The `total_loss` is backpropagated to compute gradients. If **Gradient Accumulation** is used (i.e., `gradient_accumulation` > 1), gradients are summed over several mini-batches before an optimizer step is taken, effectively simulating a larger batch size which can sometimes stabilize training. **Gradient Clipping** is then applied, where gradients are clipped to a maximum norm (`max_grad_norm`) using `torch.nn.utils.clip_grad_norm_`. This is a common technique to prevent issues with exploding gradients. Finally, the optimizer (e.g., AdamW) updates the model parameters using these accumulated and clipped gradients.

Throughout this PPO update cycle, various **Metrics are Aggregated**. These include individual loss components, the approximate KL divergence between the old and new policies, the fraction of samples where the PPO clipping was active, and explained variance. These metrics are computed and tracked using `torchmetrics`.

### 5. Logging and Checkpointing

Essential for monitoring and recovery, logging and checkpointing operations are typically performed by the rank 0 worker. After each `ppo_update` (meaning, after all epochs for a given batch of data have been processed), the aggregated training **Metrics are Logged** using `wandb_logger`. This sends the data to Weights & Biases, allowing for real-time monitoring and later analysis of the training run.

**Model Checkpointing** is performed periodically. The state of the model (its weights), the optimizer's state, and the current number of updates are saved to disk. The frequency of these saves is governed by `save_interval`, and `save_path` specifies the directory for these checkpoints. To manage disk space, **Old Checkpoint Management** ensures that older checkpoints are removed based on the `keep_interval`, for instance, by keeping only the last N checkpoints.

### 6. Reference Model Update

The `_update_ref_model` method is responsible for the synchronization of the reference model, which is used when KL regularization is active. If `enable_ref_update` is true, the weights of the `ref_model` are periodically updated to match the weights of the current trained policy model.
```{note}
This update of the reference model is often coordinated with the `broadcast_model_to_rollout_workers` calls. This ensures the reference policy doesn't lag too far behind the current policy, yet doesn't update too frequently, which could diminish its regularizing effect by making it too similar to the current policy.
```
This comprehensive cycle of data ingestion, advantage estimation, policy optimization, and model management continues for the specified `num_iterations`. The ultimate aim is to train an agent that learns to perform effectively and master complex tasks within the Minecraft environment.