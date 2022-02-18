from datetime import datetime


class TimeMeter(object):
    def __init__(self) -> None:
        super(TimeMeter, self).__init__()

        self.seconds = 0
        self.counts = 0

    def __enter__(self):
        self.start_tm = datetime.now()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.seconds += (datetime.now() - self.start_tm).total_seconds()
        self.counts += 1

    def __repr__(self) -> str:
        return f'{self.seconds / self.counts:.6f}'
