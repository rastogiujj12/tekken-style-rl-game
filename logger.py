import os, csv, datetime

class Logger:
    def __init__(self, log_dir="logs", filename_prefix="training"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(log_dir, f"{filename_prefix}_{timestamp}.csv")
        self.headers_written = False

    def log(self, **kwargs):
        """
        Append a row of key=value pairs to the CSV log file.
        Example: logger.log(episode=5, epsilon=0.9, reward=3.2, loss=0.045)
        """
        write_headers = not os.path.exists(self.filepath) or not self.headers_written
        with open(self.filepath, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(kwargs.keys()))
            if write_headers:
                writer.writeheader()
                self.headers_written = True
            writer.writerow(kwargs)

    def path(self):
        return self.filepath
