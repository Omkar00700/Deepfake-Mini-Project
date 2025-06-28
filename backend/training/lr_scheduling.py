"""
Advanced learning rate scheduling for deepfake detection
Implements layer-wise learning rate decay and OneCycleLR
"""

import tensorflow as tf
import numpy as np
import logging
import math
from typing import Tuple, List, Dict, Optional, Union, Callable

# Configure logging
logger = logging.getLogger(__name__)

class LayerwiseLRDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Layer-wise learning rate decay
    Applies different learning rates to different layers based on depth
    """
    
    def __init__(self, 
                 initial_learning_rate: float = 1e-3,
                 decay_rate: float = 0.8,
                 decay_steps: int = 1,
                 num_layers: int = 10,
                 warmup_steps: int = 0,
                 min_lr: float = 1e-6):
        """
        Initialize layer-wise learning rate decay
        
        Args:
            initial_learning_rate: Initial learning rate
            decay_rate: Rate of decay between layers
            decay_steps: Number of steps between decays
            num_layers: Number of layers in the model
            warmup_steps: Number of warmup steps
            min_lr: Minimum learning rate
        """
        super(LayerwiseLRDecay, self).__init__()
        
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.num_layers = num_layers
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        
        # Calculate layer-wise learning rates
        self.layer_learning_rates = self._calculate_layer_learning_rates()
        
        logger.info(f"Initialized layer-wise LR decay with initial_lr={initial_learning_rate}, "
                   f"decay_rate={decay_rate}, num_layers={num_layers}")
    
    def _calculate_layer_learning_rates(self) -> List[float]:
        """
        Calculate learning rates for each layer
        
        Returns:
            List of learning rates
        """
        learning_rates = []
        
        for i in range(self.num_layers):
            # Calculate decay factor
            decay_factor = self.decay_rate ** (self.num_layers - i - 1)
            
            # Calculate learning rate
            lr = max(self.initial_learning_rate * decay_factor, self.min_lr)
            
            learning_rates.append(lr)
        
        return learning_rates
    
    def __call__(self, step):
        """
        Calculate learning rate for a given step
        
        Args:
            step: Current step
            
        Returns:
            Learning rate
        """
        step = tf.cast(step, tf.float32)
        
        # Apply warmup
        if self.warmup_steps > 0:
            warmup_factor = tf.minimum(1.0, step / self.warmup_steps)
        else:
            warmup_factor = 1.0
        
        # Calculate layer index
        layer_idx = tf.minimum(
            tf.cast(step // self.decay_steps, tf.int32),
            self.num_layers - 1
        )
        
        # Get learning rate for current layer
        layer_lr = tf.gather(self.layer_learning_rates, layer_idx)
        
        return layer_lr * warmup_factor
    
    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'decay_rate': self.decay_rate,
            'decay_steps': self.decay_steps,
            'num_layers': self.num_layers,
            'warmup_steps': self.warmup_steps,
            'min_lr': self.min_lr
        }


class OneCycleLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    One Cycle Learning Rate Schedule
    Implements the 1cycle policy from the paper "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates"
    """
    
    def __init__(self, 
                 max_lr: float = 1e-3,
                 total_steps: int = 1000,
                 div_factor: float = 25.0,
                 final_div_factor: float = 1e4,
                 pct_start: float = 0.3,
                 anneal_strategy: str = 'cos'):
        """
        Initialize One Cycle LR
        
        Args:
            max_lr: Maximum learning rate
            total_steps: Total number of training steps
            div_factor: Initial learning rate divisor
            final_div_factor: Final learning rate divisor
            pct_start: Percentage of total steps spent in the increasing phase
            anneal_strategy: Annealing strategy ('cos' or 'linear')
        """
        super(OneCycleLR, self).__init__()
        
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        
        # Calculate learning rates
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        
        # Calculate step boundaries
        self.step_size_up = int(total_steps * pct_start)
        self.step_size_down = total_steps - self.step_size_up
        
        logger.info(f"Initialized OneCycleLR with max_lr={max_lr}, "
                   f"total_steps={total_steps}, pct_start={pct_start}")
    
    def __call__(self, step):
        """
        Calculate learning rate for a given step
        
        Args:
            step: Current step
            
        Returns:
            Learning rate
        """
        step = tf.cast(step, tf.float32)
        
        # Calculate learning rate
        if step < self.step_size_up:
            # Increasing phase
            if self.anneal_strategy == 'cos':
                # Cosine annealing
                cos_factor = tf.cos(
                    math.pi * (1 - step / self.step_size_up)
                )
                return self.initial_lr + (self.max_lr - self.initial_lr) * (1 + cos_factor) / 2
            else:
                # Linear annealing
                return self.initial_lr + (self.max_lr - self.initial_lr) * (step / self.step_size_up)
        else:
            # Decreasing phase
            step_down = step - self.step_size_up
            
            if self.anneal_strategy == 'cos':
                # Cosine annealing
                cos_factor = tf.cos(
                    math.pi * (step_down / self.step_size_down)
                )
                return self.final_lr + (self.max_lr - self.final_lr) * (1 + cos_factor) / 2
            else:
                # Linear annealing
                return self.max_lr - (self.max_lr - self.final_lr) * (step_down / self.step_size_down)
    
    def get_config(self):
        return {
            'max_lr': self.max_lr,
            'total_steps': self.total_steps,
            'div_factor': self.div_factor,
            'final_div_factor': self.final_div_factor,
            'pct_start': self.pct_start,
            'anneal_strategy': self.anneal_strategy
        }


class CosineAnnealingWarmRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Cosine Annealing with Warm Restarts
    Implements SGDR from the paper "SGDR: Stochastic Gradient Descent with Warm Restarts"
    """
    
    def __init__(self, 
                 initial_learning_rate: float = 1e-3,
                 first_decay_steps: int = 1000,
                 t_mul: float = 2.0,
                 m_mul: float = 1.0,
                 alpha: float = 0.0):
        """
        Initialize Cosine Annealing with Warm Restarts
        
        Args:
            initial_learning_rate: Initial learning rate
            first_decay_steps: Number of steps in the first cycle
            t_mul: Multiplicative factor for cycle length after each restart
            m_mul: Multiplicative factor for learning rate after each restart
            alpha: Minimum learning rate factor
        """
        super(CosineAnnealingWarmRestarts, self).__init__()
        
        self.initial_learning_rate = initial_learning_rate
        self.first_decay_steps = first_decay_steps
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.alpha = alpha
        
        logger.info(f"Initialized CosineAnnealingWarmRestarts with initial_lr={initial_learning_rate}, "
                   f"first_decay_steps={first_decay_steps}, t_mul={t_mul}, m_mul={m_mul}")
    
    def __call__(self, step):
        """
        Calculate learning rate for a given step
        
        Args:
            step: Current step
            
        Returns:
            Learning rate
        """
        step = tf.cast(step, tf.float32)
        
        # Calculate cycle
        cycle = tf.floor(
            tf.math.log(1.0 + step / self.first_decay_steps * (self.t_mul - 1.0)) / 
            tf.math.log(self.t_mul)
        )
        
        # Calculate cycle length
        cycle_length = self.first_decay_steps * tf.pow(self.t_mul, cycle)
        
        # Calculate cycle start
        if cycle > 0:
            cycle_start = self.first_decay_steps * (tf.pow(self.t_mul, cycle) - 1.0) / (self.t_mul - 1.0)
        else:
            cycle_start = 0.0
        
        # Calculate position in cycle
        pos_in_cycle = step - cycle_start
        
        # Calculate learning rate
        lr = self.initial_learning_rate * tf.pow(self.m_mul, cycle)
        
        # Apply cosine annealing
        cos_factor = tf.cos(
            math.pi * pos_in_cycle / cycle_length
        )
        
        return lr * (self.alpha + (1.0 - self.alpha) * (1.0 + cos_factor) / 2.0)
    
    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'first_decay_steps': self.first_decay_steps,
            't_mul': self.t_mul,
            'm_mul': self.m_mul,
            'alpha': self.alpha
        }


class AdamWOptimizer(tf.keras.optimizers.Optimizer):
    """
    AdamW optimizer with weight decay
    Implements the AdamW optimizer from the paper "Decoupled Weight Decay Regularization"
    """
    
    def __init__(self,
                learning_rate: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule] = 0.001,
                weight_decay: float = 0.01,
                beta_1: float = 0.9,
                beta_2: float = 0.999,
                epsilon: float = 1e-7,
                amsgrad: bool = False,
                name: str = "AdamW",
                **kwargs):
        """
        Initialize AdamW optimizer
        
        Args:
            learning_rate: Learning rate
            weight_decay: Weight decay factor
            beta_1: Exponential decay rate for first moment
            beta_2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
            amsgrad: Whether to use AMSGrad variant
            name: Name of the optimizer
        """
        super(AdamWOptimizer, self).__init__(name, **kwargs)
        
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("weight_decay", weight_decay)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        
        logger.info(f"Initialized AdamW optimizer with weight_decay={weight_decay}, "
                   f"beta_1={beta_1}, beta_2={beta_2}")
    
    def _create_slots(self, var_list):
        """
        Create slots for optimizer variables
        
        Args:
            var_list: List of variables
        """
        # Create slots for first and second moments
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "v")
            
            if self.amsgrad:
                self.add_slot(var, "vhat")
    
    def _prepare_local(self, var_device, var_dtype, apply_state):
        """
        Prepare local variables
        
        Args:
            var_device: Variable device
            var_dtype: Variable data type
            apply_state: Apply state
            
        Returns:
            Updated apply state
        """
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_t = tf.identity(self._get_hyper("beta_1", var_dtype))
        beta_2_t = tf.identity(self._get_hyper("beta_2", var_dtype))
        weight_decay = tf.identity(self._get_hyper("weight_decay", var_dtype))
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)
        
        lr = apply_state[(var_device, var_dtype)]["lr_t"]
        
        apply_state[(var_device, var_dtype)].update({
            "local_step": local_step,
            "beta_1_t": beta_1_t,
            "beta_2_t": beta_2_t,
            "beta_1_power": beta_1_power,
            "beta_2_power": beta_2_power,
            "weight_decay": weight_decay,
            "epsilon": tf.convert_to_tensor(self.epsilon, var_dtype)
        })
    
    def _resource_apply_dense(self, grad, var, apply_state=None):
        """
        Apply gradients to variables
        
        Args:
            grad: Gradient
            var: Variable
            apply_state: Apply state
            
        Returns:
            Operation
        """
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                       self._fallback_apply_state(var_device, var_dtype))
        
        # Get slots
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        
        # Apply weight decay
        if coefficients["weight_decay"] > 0:
            grad = grad + coefficients["weight_decay"] * var
        
        # Update moments
        m_t = coefficients["beta_1_t"] * m + (1.0 - coefficients["beta_1_t"]) * grad
        v_t = coefficients["beta_2_t"] * v + (1.0 - coefficients["beta_2_t"]) * tf.square(grad)
        
        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = tf.maximum(vhat, v_t)
            var_delta = (coefficients["lr_t"] * m_t /
                        (tf.sqrt(vhat_t) + coefficients["epsilon"]))
            vhat_update = vhat.assign(vhat_t)
        else:
            var_delta = (coefficients["lr_t"] * m_t /
                        (tf.sqrt(v_t) + coefficients["epsilon"]))
        
        # Update variable
        var_update = var.assign_sub(var_delta)
        m_update = m.assign(m_t)
        v_update = v.assign(v_t)
        
        updates = [var_update, m_update, v_update]
        
        if self.amsgrad:
            updates.append(vhat_update)
        
        return tf.group(*updates)
    
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        """
        Apply sparse gradients to variables
        
        Args:
            grad: Gradient
            var: Variable
            indices: Indices
            apply_state: Apply state
            
        Returns:
            Operation
        """
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                       self._fallback_apply_state(var_device, var_dtype))
        
        # Get slots
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        
        # Apply weight decay
        if coefficients["weight_decay"] > 0:
            grad = grad + coefficients["weight_decay"] * tf.gather(var, indices)
        
        # Update moments
        m_scaled_g_values = grad * (1 - coefficients["beta_1_t"])
        m_t = m.assign(m * coefficients["beta_1_t"])
        with tf.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)
        
        v_scaled_g_values = (grad * grad) * (1 - coefficients["beta_2_t"])
        v_t = v.assign(v * coefficients["beta_2_t"])
        with tf.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)
        
        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = vhat.assign(tf.maximum(vhat, v_t))
            v_sqrt = tf.sqrt(vhat_t)
        else:
            v_sqrt = tf.sqrt(v_t)
        
        # Update variable
        var_delta = coefficients["lr_t"] * m_t / (v_sqrt + coefficients["epsilon"])
        var_update = self._resource_scatter_add(var, indices, -var_delta)
        
        updates = [var_update, m_t, v_t]
        
        if self.amsgrad:
            updates.append(vhat_t)
        
        return tf.group(*updates)
    
    def get_config(self):
        config = super(AdamWOptimizer, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "weight_decay": self._serialize_hyperparameter("weight_decay"),
            "beta_1": self._serialize_hyperparameter("beta_1"),
            "beta_2": self._serialize_hyperparameter("beta_2"),
            "epsilon": self.epsilon,
            "amsgrad": self.amsgrad
        })
        return config


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create layer-wise learning rate decay
    lr_schedule = LayerwiseLRDecay(
        initial_learning_rate=1e-3,
        decay_rate=0.8,
        num_layers=10
    )
    
    # Create One Cycle LR
    one_cycle_lr = OneCycleLR(
        max_lr=1e-3,
        total_steps=1000
    )
    
    # Create Cosine Annealing with Warm Restarts
    cosine_lr = CosineAnnealingWarmRestarts(
        initial_learning_rate=1e-3,
        first_decay_steps=1000
    )
    
    # Create AdamW optimizer
    optimizer = AdamWOptimizer(
        learning_rate=one_cycle_lr,
        weight_decay=0.01
    )
    
    # Print learning rates
    for step in range(0, 1000, 100):
        print(f"Step {step}: LR = {one_cycle_lr(step).numpy():.6f}")