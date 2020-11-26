import albumentations as A
import cv2
import numpy as np

"""
The augmentation methods using albumentations
Note the cv2.INTER_NEAREST and cv2.BORDER_CONSTANT are necessary
"""

class _augmentation():
    def _generate_augmentation(self, mode = "distal_tubules"):
        """
        create a augmentation instance from albumentation
        args:
          mode: str, "distal_tubules" or "other", the mode of augmentation.
                      when under "distal_tubules" mode, the augmentation will
                      provide a consistent directory augmentation
        return:
          _transform: albumentations.core.composition.Compose object, taking all the transformations
        """
        assert mode in ["distal_tubules", "other"]
        if mode == "other":
            _transform =  A.Compose([A.Flip(p = 0.7),
                                    A.GridDistortion(num_steps=3,
                                                     distort_limit=0.03,
                                                     interpolation=cv2.INTER_NEAREST,
                                                     border_mode = cv2.BORDER_CONSTANT,
                                                     p = 0.3),
                                    A.Transpose(p = 0.5),
                                    A.ShiftScaleRotate(shift_limit=0.0,
                                                       scale_limit=(-0.2,0),
                                                       rotate_limit=90,
                                                       interpolation= cv2.INTER_NEAREST,
                                                       border_mode = cv2.BORDER_CONSTANT,
                                                       p = 0.5)
                                  ])
        else:
            _Vflip_flag = np.random.randint(0,2)
            _Hflip_flag = np.random.randint(0,2)
            _Transpose_flag = np.random.randint(0,2)
            _rotation = np.random.randint(0,15)
            _transform = A.Compose([
                A.VerticalFlip(p = _Vflip_flag),
                A.HorizontalFlip(p = _Hflip_flag),
                A.Transpose(p = _Transpose_flag),
                A.GridDistortion(num_steps=5,
                                distort_limit=0.05,
                                interpolation=cv2.INTER_NEAREST,
                                border_mode = cv2.BORDER_CONSTANT,
                                p = 1),
                A.ShiftScaleRotate(shift_limit=0.0,
                                  scale_limit=(-0.4,0.4),
                                  rotate_limit=(0,_rotation),
                                  interpolation= cv2.INTER_NEAREST,
                                  border_mode = cv2.BORDER_CONSTANT,
                                  p = 1)
            ])
        return _transform

    def _augment(self,image,transform):
        """
        wrapper for transformation
        transform: albumentations.core.composition.Compose object
        image: np.ndarray, the actual image

        return: np.ndarray, the transformed image from _transform
        """
        # it there's no image in this list (from utils.random_select)
        if isinstance(image, int):
            # return a smallest "valid format" zero image.
            return np.zeros((1,1,3))
        return transform(image = image)["image"]
