import io
from os import listdir
from os.path import isfile, join, splitext
from torchvision import transforms

from torch._six import string_classes
import torch as T
import torch.nn.functional as F
# from laia.data.text_image_dataset import TextImageDataset
from image_data_loader import ImageDataLoader
from utils import create_logger
from data import PadSize

import tqdm

IMAGE_EXTENSIONS = ".jpg", ".png", ".jpeg", ".pbm", ".pgm", ".ppm", ".bmp", ".tif", ".tiff", ".webp"

#_logger = log.get_logger(__name__)
logger = create_logger('Prueba', log_info_file='tmp/info.log', error_info_file='tmp/error.log')

from PIL import Image
from torch.utils import data


class ImageDataset(data.Dataset):
    def __init__(self, imgs, transform=None, pad_img=True, fixed_img_size=None):
        assert isinstance(imgs, (list, tuple))
        super(ImageDataset, self).__init__()
        self._imgs = imgs
        self._transform = transform
        self.pad_img = pad_img
        self.fixed_img_size = fixed_img_size
        if self.pad_img:
            assert isinstance(self.fixed_img_size, (list, tuple))

    def __getitem__(self, index):
        """Returns a dictionary containing the given image from the dataset.
        The image is associated with the key 'img'."""
        img = Image.open(self._imgs[index])
        if self._transform:
            img = self._transform(img)
            img_width = img.shape[-1]
        if self.pad_img:
            img = _pad(img, self.fixed_img_size)

        return {"img": img, "img_width": img_width}

    def __len__(self):
        return len(self._imgs)

def _pad(img, target_size):
    return F.pad(img, (0, target_size[1]-img.shape[-1], 0, 0))

class TextImageDataset(ImageDataset):
    def __init__(self, imgs, txts, char2num, img_transform=None, pad_img = True, fixed_img_size=None, txt_transform=None, pad_txt = False):
        super(TextImageDataset, self).__init__(imgs, img_transform, pad_img, fixed_img_size)
        assert isinstance(txts, (list, tuple))
        assert len(imgs) == len(txts)
        self._txts = txts
        self._txt_fixed_lenght = _get_max_length(txts)
        self._txt_transform = txt_transform
        self._char2num = char2num
        self._pad_txt = pad_txt

    def __getitem__(self, index):
        """
        Returns an image and its transcript from the dataset.
        :param index: Index of the item to return.
        :return: Dictionary containing the image ('img') and the transcript
            ('txt') of the image.
        """
        # Get image
        out = super(TextImageDataset, self).__getitem__(index)
        # Get transcript
        txt = self._txts[index]
        if self._txt_transform:
            txt = self._txt_transform(txt)
        # Return image and transcript
        out["txt"] = txt
        # Return also the transcript as a fully tensor
        tensor_seq = _get_sequence(txt, self._char2num)
        if self._pad_txt:
            tensor_seq = _pad_sequence(tensor_seq, self._txt_fixed_lenght)
        out["seq"] = tensor_seq
        out["txt_len"] = len(txt)
        return out

def _get_sequence(txt, char2num):
    return T.LongTensor([char2num[x] for x in txt])

def _pad_sequence(seq, fixed_length):
    return F.pad(seq, (0, fixed_length - len(seq)), mode='constant', value=0)

def _get_max_length(txts):
    return len(sorted(txts, key=len, reverse=True)[0])


class TextImageFromTextTableDataset(TextImageDataset):
    def __init__(
        self,
        txt_table,
        chars_file,
        img_dirs,
        img_transform=None,
        pad_img=True,
        fixed_img_size=None,
        txt_transform=None,
        pad_txt=False,
        img_extensions=IMAGE_EXTENSIONS,
        encoding="utf8",
    ):
        if isinstance(img_dirs, string_classes):
            img_dirs = [img_dirs]
        # First, load the transcripts and find the corresponding image filenames
        # in the given directory. Also save the IDs (basename) of the examples.
        self._ids, imgs, txts = _get_images_and_texts_from_text_table(
            txt_table, img_dirs, img_extensions, encoding=encoding
        )
        self.char2num, self.num2char = _extract_chars_from_file(chars_file)
        # Prepare dataset using the previous image filenames and transcripts.
        super(TextImageFromTextTableDataset, self).__init__(
            imgs, txts, self.char2num, img_transform, pad_img, fixed_img_size, txt_transform, pad_txt,
        )

    def __getitem__(self, index):
        """Returns the ID of the example, the image and its transcript from
        the dataset.
        Args:
          index (int): Index of the item to return.
        Returns:
          dict: Dictionary containing the example ID ('id'), image ('img') and
            the transcript ('txt') of the image.
        """
        out = super(TextImageFromTextTableDataset, self).__getitem__(index)
        out["id"] = self._ids[index]
        return out

def _extract_chars_from_file(chars_file_path, encoding='utf-8'):
    if isinstance(chars_file_path, string_classes):
        chars_file = io.open(chars_file_path, "r", encoding=encoding)
    char2num = {}
    num2char ={}
    for ind, l in enumerate(chars_file.readlines()):
        char2num[l.rstrip() if l!=' \n' else ' '] = ind
        num2char[ind] = l.rstrip() if l!=' \n' else ' '
    return char2num, num2char

def _get_valid_image_filenames_from_dir(imgs_dir, img_extensions):
    img_extensions = set(img_extensions)
    valid_image_filenames = {}
    for fname in listdir(imgs_dir):
        bname, ext = splitext(fname)
        fname = join(imgs_dir, fname)
        if isfile(fname) and ext.lower() in img_extensions:
            valid_image_filenames[bname] = fname
    return valid_image_filenames


def find_image_filename_from_id(imgid, img_dir, img_extensions):
    extensions = set(ext.lower() for ext in img_extensions)
    extensions.update(ext.upper() for ext in img_extensions)
    for ext in extensions:
        fname = join(img_dir, imgid if imgid.endswith(ext) else imgid + ext)
        if isfile(fname):
            return fname
    return None


def _load_text_table_from_file(table_file, encoding="utf-8"):
    if isinstance(table_file, string_classes):
        table_file = io.open(table_file, "r", encoding=encoding)
    for n, line in enumerate((l.split() for l in table_file), 1):
        # Skip empty lines and lines starting with #
        if not len(line) or line[0].startswith("#"):
            continue
        yield n, line[0], ' '.join(line[1:])
    table_file.close()


def _get_images_and_texts_from_text_table(
        table_file, img_dirs, img_extensions, encoding="utf8"
):
    assert len(img_dirs) > 0, "No image directory provided"
    ids, imgs, txts = [], [], []
    for _, imgid, txt in _load_text_table_from_file(table_file, encoding=encoding):
        imgid = imgid.rstrip()
        for dir in img_dirs:
            fname = find_image_filename_from_id(imgid, dir, img_extensions)
            if fname is not None:
                break
        if fname is None:
            logger.warning(
                f"No image file was found for image  'ID {imgid}', ignoring example..."
                )
            continue
        else:
            ids.append(imgid)
            imgs.append(fname)
            txts.append(txt)

    return ids, imgs, txts

if __name__ == '__main__':
    result = _load_text_table_from_file('tmp/lista_transcrip.txt')

    txt_table = 'tmp/Database/transcriptions.txt'
    img_dirs = ['tmp/Database', '/Users/aradillas/Downloads/general_data/30896/Topelius','/Users/aradillas/Downloads/general_data/30883/Goethe','/Users/aradillas/Downloads/general_data/30885/Ibsen']
    char_list_file = '/Users/aradillas/Desktop/cm_final.txt'

    height, width = 128, 2048

    dataset = TextImageFromTextTableDataset(txt_table,char_list_file, img_dirs, img_transform=transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(height),
                transforms.ToTensor(),
            ]), pad_img=False, pad_txt=False)

    batch = dataset.__getitem__(1)
    '''
    loader = T.utils.data.DataLoader(dataset=dataset,
                                         batch_size=3,
                                         shuffle=True)
    '''
    loader = ImageDataLoader(dataset=dataset,
                             batch_size=3,
                             image_channels=1,
                             image_height=None,
                             image_width=None,
                             shuffle=False)
    batch =[]
    for ele in loader:
        batch.append(ele)

    # 'application' code
    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warn message')
    logger.error('error message')
    logger.critical('critical message')
