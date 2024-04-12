class EarlyStopper:
    def __init__(
        self,
        patience: int,
        tolerance: float = 0.0,
    ):
        self.patience = patience
        self.current_epoch = 0
        self.epochs_since_best_loss = 0
        self.best_loss = float("inf")

        self._best_params = None

        self._tolerance = tolerance

    def step(self, loss: float) -> bool:
        self.current_epoch += 1

        if loss <= self.best_loss + self._tolerance:
            self.best_loss = min(loss, self.best_loss)
            self.epochs_since_best_loss = 0
        else:
            self.epochs_since_best_loss += 1

        stop = self.epochs_since_best_loss == self.patience

        return stop

    def set_best_params(self, best_params):
        self._best_params = best_params

    def get_best_params(self):
        return self._best_params

    def print_early_stop(self):
        print(f"Early stopping after {self.current_epoch} epochs.")
