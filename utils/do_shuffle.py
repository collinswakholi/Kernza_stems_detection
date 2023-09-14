from shuffle_write import ImageShuffle
import time
import os
import sys

def do_shuffle(data_folder, img_sz):
    # data_folder = "Data2/" + str(img_sz)
    data_folder = os.path.join(data_folder, str(img_sz))
    save_folder = data_folder + "_shuffled"

    shuffler = ImageShuffle(data_folder, save_folder, insert_path=True)
    shuffler.do_shuffle()
    time.sleep(1)
    shuffler.stop_shuffle()

    return data_folder, save_folder

if __name__ == "__main__":
    data_folder = sys.argv[1]
    img_sz = int(sys.argv[2])
    do_shuffle(data_folder, img_sz)
