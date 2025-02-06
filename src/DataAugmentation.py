import random
from torchvision import transforms


class DeterministicTransform:
    def __init__(self):
        self.params = {}

    def random_rotation(self, image):
        angle = random.uniform(0, 360)
        self.params["rotation"] = angle
        return transforms.functional.rotate(image, angle)

    def random_horizontal_flip(self, image):
        if random.random() > 0.5:
            self.params["flip"] = True
            return transforms.functional.hflip(image)
        self.params["flip"] = False
        return image

    def random_resized_crop(self, image):
        i, j, h, w = transforms.RandomResizedCrop(size=(512, 512)).get_params(
            image, scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3)
        )
        self.params["crop"] = (i, j, h, w)
        return transforms.functional.resized_crop(
            image,
            i,
            j,
            h,
            w,
            size=(512, 512),
            interpolation=transforms.InterpolationMode.NEAREST,
        )

    def apply_transformations(self, image):
        image = self.random_rotation(image)
        image = self.random_horizontal_flip(image)
        image = self.random_resized_crop(image)
        return image

    def apply_to_label(self, label):
        if "rotation" in self.params:
            label = transforms.functional.rotate(label, self.params["rotation"])
        if "flip" in self.params and self.params["flip"]:
            label = transforms.functional.hflip(label)
        if "crop" in self.params:
            i, j, h, w = self.params["crop"]
            label = transforms.functional.resized_crop(
                label,
                i,
                j,
                h,
                w,
                size=(512, 512),
                interpolation=transforms.InterpolationMode.NEAREST,
            )
        return label

    def generate_augmentations(self, image, label, num_augmentations=5):
        images, labels = [], []
        for _ in range(num_augmentations):
            self.params = {}
            augmented_image = self.apply_transformations(image)
            augmented_label = self.apply_to_label(label)
            images.append(augmented_image)
            labels.append(augmented_label)
        return images, labels
