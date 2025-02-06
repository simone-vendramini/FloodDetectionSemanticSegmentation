import os
import cv2
import numpy as np

from os import walk

path = os.path.join("..", "FloodNet", "FloodNet-Supervised_v1.0")
path_color = os.path.join("..", "FloodNet", "ColorMasks-FloodNetv1.0")
new_path = os.path.join("..", "FloodNet", "FloodNet-Supervised_v1.0_compressed")

color_to_label_map = {
    (0, 0, 0): 0,
    (0, 0, 255): 1,
    (120, 120, 180): 2,
    (20, 150, 160): 3,
    (140, 140, 140): 4,
    (250, 230, 61): 5,
    (255, 82, 0): 6,
    (245, 0, 255): 7,
    (0, 235, 255): 8,
    (7, 250, 4): 9,
}

label_to_color_map = {v: k for k, v in color_to_label_map.items()}


def create_all_paths():
    images_paths = {
        "train/train-org-img": [],
        "val/val-org-img": [],
        "test/test-org-img": [],
    }
    labels_paths = {
        "ColorMasks-TrainSet": [],
        "ColorMasks-ValSet": [],
        "ColorMasks-TestSet": [],
    }

    for key, _ in images_paths.items():
        tmp_path = path + key + "/"
        for _, _, filenames in walk(tmp_path):
            images_paths[key].extend([tmp_path + filename for filename in filenames])
            break

    for key, _ in labels_paths.items():
        tmp_path = path_color + key + "/"
        for _, _, filenames in walk(tmp_path):
            labels_paths[key].extend([tmp_path + filename for filename in filenames])
            break

    return images_paths, labels_paths


def resize_all_image(paths, real=True):
    for key, _ in paths.items():
        if not os.path.exists(new_path + key):
            os.makedirs(new_path + key)

        for i in range(len(paths[key])):
            head_tail = os.path.split(paths[key][i])
            img = cv2.imread(paths[key][i])
            if real:
                img = cv2.resize(img, (1024, 1024))
                cv2.imwrite(new_path + key + "/" + head_tail[1], img)
            else:
                map_color_to_label(np.array(img), new_path + key + "/" + head_tail[1])


def map_color_to_label(img, path):
    labeled_image = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = tuple(img[i, j])
            labeled_image[i, j] = color_to_label_map.get(pixel, 0)

    cv2.imwrite(path, labeled_image)


if __name__ == "__main__":
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    image_paths, label_paths = create_all_paths()

    print("Compressing images...")
    resize_all_image(image_paths, True)
    print("Compressing labels...")
    resize_all_image(label_paths, False)
