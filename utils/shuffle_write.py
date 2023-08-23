import glob
import os
import numpy as np
import random
import shutil
import threading

class ImageShuffle:
    def __init__(self, folder_name, save_folder, insert_path=False):
        self.folder_name = folder_name
        self.save_folder = save_folder
        self.insert_path = insert_path
        self.ratio = [0.7, 0.2, 0.1]
        self.seed_number = np.random.randint(0,1000)
        self.stop_event = threading.Event()
        self.shuffle_thread = None
        
    @staticmethod
    def _copy(image_list, label_list, image_folder, label_folder):
        for image, label in zip(image_list, label_list):
            image_name = os.path.basename(image)
            label_name = os.path.basename(label)
            
            if image_name.split(".jpg")[0] != label_name.split(".txt")[0]:
                print("Image and label names do not match")
                print("Image name =", image_name)
                print("Label name =", label_name)
            else:
                shutil.copy(image, image_folder)
                shutil.copy(label, label_folder)
            
    # @staticmethod
    # def find_similar(str, list):
    #     # find the 
            
    def do_shuffle(self):
        if self.shuffle_thread is not None and self.shuffle_thread.is_alive():
            print("Thread is alive, shuffling still in progress...")
            return
        self.stop_event.clear()
        self.shuffle_thread = threading.Thread(target=self.shuffle_write)
        self.shuffle_thread.start()
        
    def stop_shuffle(self):
        if self.shuffle_thread is None or not self.shuffle_thread.is_alive():
            print("No shuffling is currently in progress.")
            return

        self.stop_event.set()
        self.shuffle_thread.join()
        print("Shuffling has been stopped.")

    def shuffle_write(self):
        shutil.rmtree(self.save_folder, ignore_errors=True)
        os.makedirs(self.save_folder)
        os.makedirs(os.path.join(self.save_folder, "train", "images"))
        os.makedirs(os.path.join(self.save_folder, "train", "labels"))
        os.makedirs(os.path.join(self.save_folder, "valid", "images"))
        os.makedirs(os.path.join(self.save_folder, "valid", "labels"))
        os.makedirs(os.path.join(self.save_folder, "test", "images"))
        os.makedirs(os.path.join(self.save_folder, "test", "labels"))

        list_of_files = glob.glob(self.folder_name + '/*')
        list_of_files = [x for x in list_of_files if os.path.isfile(x)]

        for file in list_of_files:
            shutil.copy(file, self.save_folder)

        if self.insert_path:
            with open(os.path.join(self.save_folder, "data.yaml"), "r") as f:
                lines = f.readlines()

            line0 = lines[0]
            line0 = line0.replace("../", "")
            lines[0] = "path: ../" + self.save_folder + "\n" + line0

            lines[1] = lines[1].replace("../", "")
            lines[2] = lines[2].replace("../", "")
            lines[3] = lines[3].replace("../", "")

            with open(os.path.join(self.save_folder, "data.yaml"), "w") as f:
                f.writelines(lines)
            
        image_folder = os.path.join(self.folder_name, "train", "images")
        labels_folder = os.path.join(self.folder_name, "train", "labels")

        images_list = glob.glob(os.path.join(image_folder, "*.jpg"))
        labels_list = glob.glob(os.path.join(labels_folder, "*.txt"))

        num_images = len(images_list)

        print("Seed number = "+str(self.seed_number))

        random.seed(self.seed_number)
        shuffled_images_list = random.sample(images_list, num_images)
        random.seed(self.seed_number)
        shuffled_labels_list = random.sample(labels_list, num_images)

        # split the list into train, valid, and test set
        split = self.ratio

        train_split = int(split[0]*num_images)
        valid_split = int((split[0]+split[1])*num_images)
        test_split = int((split[0]+split[1]+split[2])*num_images)

        train_images = shuffled_images_list[0:train_split]
        valid_images = shuffled_images_list[train_split:valid_split]
        test_images = shuffled_images_list[valid_split:test_split]

        train_labels = shuffled_labels_list[0:train_split]
        valid_labels = shuffled_labels_list[train_split:valid_split]
        test_labels = shuffled_labels_list[valid_split:test_split]

        self._copy(
            train_images, 
            train_labels, 
            os.path.join(self.save_folder, "train", "images"), 
            os.path.join(self.save_folder, "train", "labels")
            )
        
        self._copy(
            valid_images,
            valid_labels,
            os.path.join(self.save_folder, "valid", "images"),
            os.path.join(self.save_folder, "valid", "labels")
            )
        
        self._copy(
            test_images,
            test_labels,
            os.path.join(self.save_folder, "test", "images"),
            os.path.join(self.save_folder, "test", "labels")
            )
        
        print("\nImages have been saved to '"+self.save_folder+"'")
        print("\nDone shuffling...")
 