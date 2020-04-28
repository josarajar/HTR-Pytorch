import torch as T
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image

class PadSize(transforms.Pad):
    """Pad the given PIL Image to get an image of the size provided in the "size" value. Is an adaptation
        of the torchvision.transforms.Pad() class.

        Args:
            size (int or tuple): Desired size of output image. If a single int is provided the
                output image will be a square. If tuple of length 2 is provided this corresponds to
                the (height, width) tuple.
            fill (int or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
                length 3, it is used to fill R, G, B channels respectively.
                This value is only used when the padding_mode is constant
            padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
                Default is constant.

                - constant: pads with a constant value, this value is specified with fill

                - edge: pads with the last value at the edge of the image

                - reflect: pads with reflection of image without repeating the last value on the edge

                    For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                    will result in [3, 2, 1, 2, 3, 4, 3, 2]

                - symmetric: pads with reflection of image repeating the last value on the edge

                    For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                    will result in [2, 1, 1, 2, 3, 4, 4, 3]
        """
    def __init__(self, size, fill=0, padding_mode='constant'):
        super().__init__(padding=size, fill=fill, padding_mode=padding_mode)

        self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return F.pad(img, (0, 0, self.size[1] - img.size[0], self.size[0] - img.size[1]), self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'. \
            format(self.fill, self.padding_mode)



traindir = '/Users/aradillas/processed_images'


train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(80),
            PadSize(size=(100,80)),
            transforms.ToTensor(),
        ]))

loader = T.utils.data.DataLoader(dataset=train_dataset,
                                     batch_size=10,
                                     shuffle=False)

if __name__ == '__main__':
    for x,y in loader:
        Image.fromarray(x[0,0,:,:].numpy()*255).show()

    print("Fin del programa")
'''
img = Image.open('/Users/aradillas/processed_images/BHIC_5117-085-0027_boxes.jpg')
padder = PadSize(size=(1200,1000))
img2 = padder(img)
'''