from PIL import Image
import os

class Pgm(object):
    def is_pgm_file(self,in_path):
        if not os.path.exists(in_path):
            return False
        if in_path is not str and not in_path.endswith('.pgm'):
            return False
        return True

    def pgm_to_jpg(self,in_path,out_path):
        if self.is_pgm_file(in_path)==False:
            raise Exception(f'{in_path}不是一个pgm文件')
        else:
            img=Image.open(in_path)
            img.save(out_path)