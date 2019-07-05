from tensorboardX import SummaryWriter


class PyTorchSummaryWriter(SummaryWriter):
    def __init__(self, comment, flush_secs=5):
        super().__init__(comment=comment, flush_secs=flush_secs)

    def get_log_dir(self):
        return self.log_dir

    def close(self):
        self.export_scalars_to_json("./all_scalars.json")
        super(PyTorchSummaryWriter, self).close()
