import yadisk
import getpass
import os


class CloudSaver:
    def __init__(self, ya_path):
        self.ya_path = ya_path
        ya_token = getpass.getpass("Token:")
        self.ya_disk = yadisk.YaDisk(token=ya_token)
        assert self.ya_disk.exists(self.ya_path), f"Папка с именем {self.ya_path} отсутствует в облаке"

    def save(self, checkpoint_file_name, tensorboard_dir):
        ya_checkpoint_file_name = os.path.join(self.ya_path, os.path.basename(checkpoint_file_name))
        self.ya_disk.upload(checkpoint_file_name, ya_checkpoint_file_name)

        base_dir = os.path.basename(tensorboard_dir)
        for item in os.walk(tensorboard_dir):
            ya_tensorboard_dir = os.path.join(self.ya_path, base_dir)
            if not self.ya_disk.exists(ya_tensorboard_dir):
                self.ya_disk.mkdir(ya_tensorboard_dir)

            for filename in item[2]:
                local_filename = os.path.join(tensorboard_dir, filename)
                ya_filename = os.path.join(ya_tensorboard_dir, filename)
                if self.ya_disk.exists(ya_filename):
                    meta = self.ya_disk.get_meta(ya_filename)
                    size = meta['size']
                    stat = os.stat(local_filename)
                    if size != stat.st_size:
                        self.ya_disk.upload(local_filename, ya_filename, overwrite=True)
                else:
                    self.ya_disk.upload(local_filename, ya_filename, overwrite=True)

            break

