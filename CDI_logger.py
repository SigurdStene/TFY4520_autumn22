import os
import time

class Logger():
    def __init__(self, filename_prefix):
        self.filename_prefix = filename_prefix
        self.log = []
        self.log.append("Log created at {}".format(time.asctime()))
        self.log.append("")

        if not os.path.isdir(os.path.join(os.getcwd(), "Log")):
            os.mkdir(os.path.join(os.getcwd(), "Log"))
            time.sleep(1)

        self.save_path = os.path.join(os.getcwd(), "Log")   

    def add_log(self, log):
        self.log.append(log)

    def print_log(self):
        for log in self.log:
            print(log)

    def save_log(self):
        with open(os.path.join(self.save_path, self.filename_prefix + ".txt"), "w") as f:
            for log in self.log:
                f.write(log + "\n")
