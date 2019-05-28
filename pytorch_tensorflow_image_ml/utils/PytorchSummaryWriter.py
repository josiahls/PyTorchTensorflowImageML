from tensorboardX import SummaryWriter


class PyTorchSummaryWriter(SummaryWriter):
    def __init__(self, comment, flush_secs=5):
        super().__init__(comment=comment, flush_secs=flush_secs)
