import argparse

class arguments():
    def __init__(self):
        self.argparser = argparse.ArgumentParser()
        self.initialize()
    
    def initialize(self):
        self.argparser.add_argument('--frame_dir', type=str, default='frame', help='path to frames')
        self.argparser.add_argument('--img_save_dir', type=str, default='results', help='path to storage generated feature maps if needed')
        self.argparser.add_argument('--n_epoch', type=int, default=1, help='number of epochs')
        self.argparser.add_argument('--n_threads', type=int, default=1, help='number of threads for dataloader')
        self.argparser.add_argument('--batch_size', type=int, default=1, help='just batch size')
        self.argparser.add_argument('--is_shuffle', type=bool, default=False, help='Do shuffle during loading data or not')

    def parse(self):
        self.args = self.argparser.parse_args()
        return self.args
