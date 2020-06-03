############################# IMPORTS #############################

from PIL import Image
from torchvision.datasets import FashionMNIST

###################################################################


class CustomFashionMNIST(FashionMNIST):
    '''
    Extension of pytorch's FashionMNIST class to distinguish between classes.
    '''

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        transform = self.__get_transformation__(target)
        if transform is not None:
            img = transform(img)

        if self.target_transform is not None:
            exit(0)
            target = self.target_transform(target)
        return img, target

    def __get_transformation__(self, label):
        '''
        Returns the transformation if it is inside of a dictionary with labels as keys.
        :param label: A tensor variable.
        :return: a Compose transformation
        '''
        if isinstance(self.transform, dict):
            return self.transform[label]
        else:
            return self.transform
