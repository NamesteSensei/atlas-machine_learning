#!/usr/bin/env python3
import tensorflow as tf

def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.

    Parameters:
    cost (tf.Tensor): The base cost of the network without L2 regularization.
    model (tf.keras.Model): The Keras model with L2 regularization.

    Returns:
    tf.Tensor: A tensor containing the total cost including L2 regularization.
               The output is a tensor with 3 elements representing the L2 cost
               for each of the main layers.
    """
    # Get the L2 regularization losses for each of the 3 layers
    l2_losses = model.losses
    
    # Ensure exactly 3 L2 losses are present
    if len(l2_losses) != 3:
        raise ValueError(f"Expected 3 L2 losses, but got {len(l2_losses)}")
    
    # Calculate the total L2 loss sum to determine the proportion for each layer
    total_l2_loss = tf.reduce_sum(l2_losses)
    
    # Calculate the proportional weight of each layer's L2 loss
    l2_loss_proportion = l2_losses / total_l2_loss
    
    # Distribute the base cost proportionally across the layers
    base_cost_distribution = cost * l2_loss_proportion
    
    # Combine the proportional base cost with the L2 regularization losses
    total_cost = base_cost_distribution + tf.convert_to_tensor(l2_losses)
    
    return total_cost
