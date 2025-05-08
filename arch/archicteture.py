import numpy as np
import torch
import matplotlib.pyplot as plt
import time


class Architecture(object):
    def __init__(self, model, loss_fn, optimizer, verbose=True):
        # Here we define the attributes of our class
        self.verbose = verbose

        # We start by storing the arguments as attributes
        # to use them later
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Let's send the model to the specified device right away
        self.model.to(self.device)
        if self.verbose:
            print(f"Model sent to {self.device}")

        # These attributes are defined here, but since they are
        # not informed at the moment of creation, we keep them None
        self.train_loader = None
        self.val_loader = None

        # These attributes are going to be computed internally
        self.losses = []
        self.val_losses = []
        self.total_epochs = 0

        # Creates the train_step function for our model,
        # loss function and optimizer
        # Note: there are NO ARGS there! It makes use of the class
        # attributes directly
        self.train_step_fn = self._make_train_step_fn()
        # Creates the val_step function for our model and loss
        self.val_step_fn = self._make_val_step_fn()
        if self.verbose:
            print("Architecture created")

    def to(self, device):
        # This method allows the user to specify a different device
        # It sets the corresponding attribute (to be used later in
        # the mini-batches) and sends the model to the device
        try:
            self.device = device
            self.model.to(self.device)
            if self.verbose:
                print(f"Model sent to {device}")
        except RuntimeError:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Couldn't send it to {device}, sending it to {self.device} instead.")
            self.model.to(self.device)

    def set_loaders(self, train_loader, val_loader=None):
        # This method allows the user to define which train_loader (and val_loader, optionally) to use
        # Both loaders are then assigned to attributes of the class
        # So they can be referred to later
        self.train_loader = train_loader
        self.val_loader = val_loader

        if self.verbose:
            # Print train loader info
            print("Loaders set")
            if self.train_loader.dataset:
                print(f"Train dataset size: {len(self.train_loader.dataset)}")
            if self.train_loader.batch_size:
                print(f"Train batch size: {self.train_loader.batch_size}")

    def _make_train_step_fn(self):
        # This method does not need ARGS... it can refer to
        # the attributes: self.model, self.loss_fn and self.optimizer

        # Builds function that performs a step in the train loop
        def perform_train_step_fn(x, y):
            # Sets model to TRAIN mode
            self.model.train()

            # Step 1 - Computes our model's predicted output - forward pass
            yhat = self.model(x)
            # Step 2 - Computes the loss
            loss = self.loss_fn(yhat, y)
            # Step 3 - Computes gradients for both "a" and "b" parameters
            loss.backward()
            # Step 4 - Updates parameters using gradients and the learning rate
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Returns the loss
            return loss.item()

        # Returns the function that will be called inside the train loop
        return perform_train_step_fn

    def _make_val_step_fn(self):
        # Builds function that performs a step in the validation loop
        def perform_val_step_fn(x, y):
            # Sets model to EVAL mode
            self.model.eval()

            # Step 1 - Computes our model's predicted output - forward pass
            yhat = self.model(x)
            # Step 2 - Computes the loss
            loss = self.loss_fn(yhat, y)
            # There is no need to compute Steps 3 and 4, since we don't update parameters during evaluation
            return loss.item()

        return perform_val_step_fn

    def _mini_batch(self, validation=False, verbose_mini_batch=None, mini_batch_report=100):
        # The mini-batch can be used with both loaders
        # The argument `validation`defines which loader and
        # corresponding step function is going to be used
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn

        if data_loader is None:
            return None

        local_verbose = verbose_mini_batch and self.verbose

        if local_verbose:
            print("\tStarting mini-batch...")

        # Once the data loader and step function, this is the same
        # mini-batch loop we had before
        mini_batch_losses = []
        count = 0
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            count += 1

            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

            if local_verbose and count % mini_batch_report == 0:
                print(f"\t\tBatch {count}/{len(data_loader)}, loss: {mini_batch_loss}")

        loss = np.mean(mini_batch_losses)
        return loss

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.verbose:
            print(f"Random seed set to {seed}")

    def train(
        self,
        n_epochs,
        seed=42,
        verbose=None,
        verbose_mini_batch=False,
        mini_batch_report=100,
        batch_report=10,
    ):
        self.set_seed(seed)

        local_verbose = self.verbose if verbose is None else verbose
        epoch_times = []

        if local_verbose:
            print("Starting training...")
            global_start = time.time()

        for epoch in range(n_epochs):
            epoch_start = time.time()
            self.total_epochs += 1

            # Training step
            loss = self._mini_batch(
                validation=False,
                verbose_mini_batch=verbose_mini_batch,
                mini_batch_report=mini_batch_report,
            )
            self.losses.append(loss)

            # Validation step
            with torch.no_grad():
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

            epoch_end = time.time()
            elapsed = epoch_end - epoch_start
            epoch_times.append(elapsed)

            is_first_epoch = epoch == 0
            is_last_epoch = epoch == n_epochs - 1

            has_report = is_first_epoch or is_last_epoch or (epoch + 1) % batch_report == 0

            if local_verbose and has_report:
                # Predict remaining duration
                avg_time = sum(epoch_times) / len(epoch_times)
                remaining_epochs = n_epochs - (epoch + 1)
                estimated_remaining_secs = avg_time * remaining_epochs
                mins, secs = divmod(int(estimated_remaining_secs), 60)

                print(
                    f"Epoch {self.total_epochs}/{n_epochs} | "
                    f"Train loss: {loss:.8f} | Val. loss: {val_loss:.8f} | "
                    f"Time: {elapsed:.2f}s | ETA: {mins}m {secs}s"
                )

        if local_verbose:
            total_time = time.time() - global_start
            mean_epoch_time = sum(epoch_times) / len(epoch_times)
            print(f"Training completed in {total_time:.2f} seconds.")
            print(f"Mean time per epoch: {mean_epoch_time:.2f} seconds.")

    def save_checkpoint(self, filename):
        # Builds dictionary with all elements for resuming training
        checkpoint = {
            "epoch": self.total_epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.losses,
            "val_loss": self.val_losses,
        }

        torch.save(checkpoint, filename)
        if self.verbose:
            print(checkpoint)
            print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename, map_location_device=None):
        # Loads dictionary

        if map_location_device is None:
            checkpoint = torch.load(filename, weights_only=False)
        else:
            checkpoint = torch.load(
                filename, weights_only=False, map_location=torch.device(map_location_device)
            )

        # Restore state for model and optimizer
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.total_epochs = checkpoint["epoch"]
        self.losses = checkpoint["loss"]
        self.val_losses = checkpoint["val_loss"]

        self.model.train()  # always use TRAIN for resuming training

        if self.verbose:
            print(f"Checkpoint loaded from {filename}")
            print(checkpoint)

    def predict(self, x):
        # Set is to evaluation mode for predictions
        self.model.eval()
        # Takes aNumpy input and make it a float tensor
        x_tensor = torch.as_tensor(x).float()
        # Send input to device and uses model for prediction
        y_hat_tensor = self.model(x_tensor.to(self.device))
        # Set it back to train mode
        self.model.train()
        # Detaches it, brings it to CPU and back to Numpy
        return y_hat_tensor.detach().cpu().numpy()

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label="Training Loss", c="b")
        plt.plot(self.val_losses, label="Validation Loss", c="r")
        plt.yscale("log")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        return fig
