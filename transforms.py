from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as F

class Resize(transforms.Resize):

    def __init__(self, size, interpolation=Image.LANCZOS):
        super(Resize, self).__init__(size, interpolation)

    def __call__(self, img):
        if isinstance(self.size, int):
            self.adaptative_size = (self.size, round(self.size * img.size[0] / img.size[1]))
        return F.resize(img, self.adaptative_size, self.interpolation)
