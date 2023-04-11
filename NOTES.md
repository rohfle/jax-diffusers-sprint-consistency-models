# TODO
- integrate ema model callback into training loop
- complete UNet class
- dataloading




## Internal notes

UNet2DModel - https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d.py#L38


# UNet is type of neural network architecture used for images
class UNet(UNet2DModel):
    def forward(self, x): return super().forward(*x).sample



unet = UNet(in_channels=1, out_channels=1, block_out_channels=(16, 32, 64, 64), norm_num_groups=8)
model = ConsistencyUNet(0.002, unet)



    Initialize the model: Initialize the weights and biases of the model randomly or with pre-trained weights.

    Prepare the data: Load the training dataset and preprocess it as necessary. This may include normalization, data augmentation, and splitting the dataset into batches.

    Define the loss function: Choose an appropriate loss function that measures the difference between the model's predictions and the true values.

    Define the optimizer: Choose an appropriate optimizer that updates the model parameters based on the gradients of the loss function.

    Training loop: Loop over the dataset for a fixed number of epochs, or until a convergence criterion is met. For each batch of data:

        Compute the model's predictions on the batch of data.

        Compute the loss function on the model's predictions and the true labels.

        Compute the gradients of the loss function with respect to the model parameters.

        Use the optimizer to update the model parameters based on the gradients.

    Evaluation: After training, evaluate the model on a validation dataset to check its performance on unseen data.

    Save the model: Save the trained model's parameters to disk for later use.




