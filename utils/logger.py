from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    _instance = None  # Singleton instance

    def __new__(cls, log_dir="runs", comment="", filename_suffix=""):
        if cls._instance is None:
            cls._instance = super(TensorBoardLogger, cls).__new__(cls)
            cls._instance.writer = SummaryWriter(log_dir=log_dir, comment=comment, filename_suffix=filename_suffix)
            cls._instance.step = 0  # Global step counter
            cls._instance.log_dir = log_dir
            cls._instance.initialized = True
            print(f"TensorBoardLogger initialized in {log_dir}") # confirmation
        return cls._instance

    @classmethod
    def instance(cls, log_dir="runs", comment="", filename_suffix=""):
        return cls(log_dir, comment, filename_suffix) # Calls __new__

    def log_scalar(self, tag, value, step=None):
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag, values, step=None, bins='tensorflow'):
        if step is None:
            step = self.step
        self.writer.add_histogram(tag, values, step, bins=bins)

    def log_image(self, tag, img_tensor, step=None, dataformats='CHW'):
        if step is None:
            step = self.step
        self.writer.add_image(tag, img_tensor, step, dataformats=dataformats)

    def log_graph(self, model, input_to_model):
        self.writer.add_graph(model, input_to_model)


    def increment_step(self):
        self.step += 1

    def close(self):
        self.writer.close()
        self.initialized = False
        print("TensorBoardLogger closed.")