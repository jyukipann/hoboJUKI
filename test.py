import pathlib
from traceback import print_tb
import cv2

dir_paths = pathlib.Path(r"J:\doc\MIRU2022\MIRU2022\figs")
print(dir_paths)
file_paths = dir_paths.glob('*.png')

def f(img):
    # なんかする
    return img

for path in file_paths:
    # img = cv2.imread(str(path))
    # img = f(img)
    print(path.name)

    # cv2.waitKey(0)