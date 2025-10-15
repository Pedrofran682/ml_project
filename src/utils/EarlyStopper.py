class EarlyStopper:
    """A utility to stop model training early if a monitored metric stops improving.

    This helps prevent overfitting by halting the training process once the model's
    performance on a validation set ceases to improve for a specified number of
    consecutive epochs.

    Attributes:
        patience (int): The number of epochs to wait for improvement before
            triggering a stop.
        min_delta (float): The minimum change in the monitored metric to be
            considered an improvement.
        counter (int): The current count of epochs without improvement.
        min_validation_loss (float): The lowest validation loss observed so far.
    """

    def __init__(self, patience=1, min_delta=1e-5):
        """Initializes the EarlyStopper.

        Args:
            patience (int, optional): The number of epochs with no improvement
                after which training will be stopped. Defaults to 1.
            min_delta (float, optional): The minimum change in the monitored
                quantity to qualify as an improvement. This is meant to prevent
                stopping for trivial improvements. Defaults to 0.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        """Checks if the training process should be stopped.

        This method should be called once per epoch. It evaluates the current
        validation loss against the best loss seen so far and updates its
        internal counter.

        Args:
            validation_loss (float): The validation loss from the current epoch.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False