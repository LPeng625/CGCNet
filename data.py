"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import torch
import torch.utils.data as data
from torch.autograd import Variable as V
from PIL import Image

import cv2
import numpy as np
import os
import Constants, Constants2


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        # image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask


def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image, mask


def default_loader(img_path, mask_path):
    img = cv2.imread(img_path)
    # print("img:{}".format(np.shape(img)))
    img = cv2.resize(img, (1024, 1024))

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask = 255. - cv2.resize(mask, (1024, 1024))

    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)

    mask = np.expand_dims(mask, axis=2)
    #
    # print(np.shape(img))
    # print(np.shape(mask))

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    # mask = np.array(mask, np.float32).transpose(2,0,1)/255.0
    # mask[mask >= 0.5] = 1
    # mask[mask <= 0.5] = 0
    # mask[mask <= 0.5] = 1

    # mask = abs(mask-1)
    return img, mask


def default_DRIVE_loader(img_path, mask_path):
    img = cv2.imread(img_path)
    if 'Massachusetts' in img_path:
        image_size = Constants2.Image_size
    else:
        image_size = Constants.Image_size
    img = cv2.resize(img, image_size)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # mask = np.array(Image.open(mask_path))

    # 将标签输入改成跟论文一样的格式
    mask[mask > 0] = 255
    # mask = cv2.resize(mask, (1024, 1024))[:, :, 0]
    mask = cv2.resize(mask, image_size)

    # print(img.shape, mask.shape)

    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)
    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6

    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    # mask = abs(mask-1)
    # label1 = Image.fromarray(np.uint8(mask))  # 输出标签
    # label1.putpalette([0, 0, 0, 128, 0, 0])
    # label1.save("./test/" + 'mask' + '.png')  # 存下图片

    # cv2.imwrite("./test/" + 'label' + '.png', mask)  # 存下图片
    #
    # image1 = Image.fromarray(np.uint8(img))
    # image1.save("./test/" + 'image' + '.png')  # 存下图片

    return img, mask


removeImage = ['17428975_15.tiff', '20429005_15.tiff', '26278705_15.tiff', '20728870_15.tiff', '17578795_15.tiff', '24778885_15.tiff', '23279050_15.tiff', '26428690_15.tiff', '18928780_15.tiff', '25379290_15.tiff', '24179170_15.tiff', '10228795_15.tiff', '24329005_15.tiff', '23129170_15.tiff', '23878930_15.tiff', '10528795_15.tiff', '12478810_15.tiff', '17428735_15.tiff', '18928720_15.tiff', '16978945_15.tiff', '25529170_15.tiff', '17578840_15.tiff', '21478870_15.tiff', '25229170_15.tiff', '17728810_15.tiff', '19678630_15.tiff', '10378795_15.tiff', '10978915_15.tiff', '19978645_15.tiff', '23429050_15.tiff', '18478945_15.tiff', '25529290_15.tiff', '26578690_15.tiff', '18628915_15.tiff', '27028675_15.tiff', '11128645_15.tiff', '12628795_15.tiff', '21928960_15.tiff', '19078750_15.tiff', '16978870_15.tiff', '18328885_15.tiff', '26428705_15.tiff', '17128960_15.tiff', '21478945_15.tiff', '20579005_15.tiff', '25828765_15.tiff', '23128870_15.tiff', '20279005_15.tiff', '17578990_15.tiff', '24629305_15.tiff', '27178705_15.tiff', '16978885_15.tiff', '24329050_15.tiff', '23129125_15.tiff', '25828780_15.tiff', '21328870_15.tiff', '12478720_15.tiff', '16828900_15.tiff', '24029185_15.tiff', '23879200_15.tiff', '26578675_15.tiff', '18478885_15.tiff', '20878990_15.tiff', '16378855_15.tiff', '25229185_15.tiff', '25379185_15.tiff', '17728960_15.tiff', '10078720_15.tiff', '10678810_15.tiff', '15628840_15.tiff', '17728855_15.tiff', '24778870_15.tiff', '17728720_15.tiff', '26429275_15.tiff', '23729230_15.tiff', '16228960_15.tiff', '23129155_15.tiff', '23579140_15.tiff', '19828915_15.tiff', '23878915_15.tiff', '18328810_15.tiff', '10828645_15.tiff', '23729110_15.tiff', '16228885_15.tiff', '11278885_15.tiff', '21178975_15.tiff', '25978750_15.tiff', '21778930_15.tiff', '21779080_15.tiff', '12028660_15.tiff', '16378825_15.tiff', '18178975_15.tiff', '23429065_15.tiff', '16228825_15.tiff', '18478960_15.tiff', '26878750_15.tiff', '11278900_15.tiff', '18628930_15.tiff', '21928945_15.tiff', '22679080_15.tiff', '10078750_15.tiff', '24028555_15.tiff', '19078765_15.tiff', '10678870_15.tiff', '17128945_15.tiff', '21028975_15.tiff', '21628930_15.tiff', '23728765_15.tiff', '19228735_15.tiff', '15628915_15.tiff', '25229155_15.tiff', '21329035_15.tiff', '15928960_15.tiff', '16228900_15.tiff', '10978645_15.tiff', '26878675_15.tiff', '19978660_15.tiff', '23129440_15.tiff', '18478720_15.tiff', '22379440_15.tiff', '25829290_15.tiff', '18628795_15.tiff', '26278810_15.tiff', '20879005_15.tiff', '24778780_15.tiff', '22529380_15.tiff', '24179110_15.tiff', '11428945_15.tiff', '10678885_15.tiff', '21629065_15.tiff', '23129110_15.tiff', '25979230_15.tiff', '18328720_15.tiff', '10228660_15.tiff', '17728705_15.tiff', '23279080_15.tiff', '19528645_15.tiff', '24478990_15.tiff', '99238675_15.tiff', '21779065_15.tiff', '16228945_15.tiff', '24628780_15.tiff', '24029110_15.tiff', '24629155_15.tiff', '22079080_15.tiff', '10978840_15.tiff', '22379380_15.tiff', '22229440_15.tiff', '25979290_15.tiff', '18778795_15.tiff', '22529440_15.tiff', '10678825_15.tiff', '24178480_15.tiff', '24479110_15.tiff', '10978855_15.tiff', '23429035_15.tiff', '21628960_15.tiff', '15628870_15.tiff', '19078735_15.tiff', '22529455_15.tiff', '23428510_15.tiff', '12178705_15.tiff', '23728600_15.tiff', '23279095_15.tiff', '25079155_15.tiff', '17278750_15.tiff', '16978930_15.tiff', '22829485_15.tiff', '19978630_15.tiff', '99238660_15.tiff', '23129410_15.tiff', '18028810_15.tiff', '26728780_15.tiff', '18628720_15.tiff', '19978945_15.tiff', '23878570_15.tiff', '12028690_15.tiff', '23578495_15.tiff', '17878855_15.tiff', '19828630_15.tiff', '18178810_15.tiff', '23878945_15.tiff', '18778720_15.tiff', '23579110_15.tiff', '12028705_15.tiff', '17428750_15.tiff', '24478765_15.tiff', '16828930_15.tiff', '22229065_15.tiff', '21778960_15.tiff', '21328975_15.tiff', '26128810_15.tiff', '24479095_15.tiff', '22529395_15.tiff', '24928900_15.tiff', '24028480_15.tiff', '23279035_15.tiff', '17878960_15.tiff', '18328975_15.tiff', '10828855_15.tiff', '24329155_15.tiff', '15628900_15.tiff', '24328765_15.tiff', '10078660_15.tiff', '17278870_15.tiff', '19828885_15.tiff', '21328990_15.tiff', '23729245_15.tiff', '16228915_15.tiff', '21329020_15.tiff', '23728480_15.tiff', '26578795_15.tiff', '19978675_15.tiff', '23129455_15.tiff', '24028795_15.tiff', '18028960_15.tiff', '22529080_15.tiff', '18478975_15.tiff', '11878795_15.tiff', '22229080_15.tiff', '24479155_15.tiff', '22979470_15.tiff', '11278915_15.tiff', '24178975_15.tiff', '15628825_15.tiff', '24329095_15.tiff', '26128720_15.tiff', '18478795_15.tiff', '24178540_15.tiff', '12028675_15.tiff', '23878585_15.tiff', '26728765_15.tiff', '23129065_15.tiff', '24329110_15.tiff', '22379080_15.tiff', '23129140_15.tiff', '23579155_15.tiff', '21479065_15.tiff', '26428795_15.tiff', '17878810_15.tiff', '15628885_15.tiff', '17428870_15.tiff', '19978885_15.tiff', '23428525_15.tiff', '21628870_15.tiff', '24328870_15.tiff', '15628960_15.tiff', '17278765_15.tiff', '23129425_15.tiff', '25978795_15.tiff', '18028825_15.tiff', '10978945_15.tiff', '21329050_15.tiff', '21178885_15.tiff', '21929080_15.tiff', '12478795_15.tiff', '10228765_15.tiff', '19228750_15.tiff', '18178885_15.tiff', '21628945_15.tiff', '17728975_15.tiff', '20729005_15.tiff', '21028900_15.tiff', '15628855_15.tiff', '10078735_15.tiff', '22529470_15.tiff', '21778945_15.tiff', '25978735_15.tiff', '24478900_15.tiff', '20578870_15.tiff', '17728795_15.tiff', '16378840_15.tiff', '10828870_15.tiff', '23428585_15.tiff', '23278885_15.tiff', '18178825_15.tiff', '23878960_15.tiff', '17878705_15.tiff', '23579125_15.tiff', '12628810_15.tiff', '23428900_15.tiff', '15778960_15.tiff', '23879215_15.tiff', '16828915_15.tiff', '25679275_15.tiff', '11278645_15.tiff', '17428780_15.tiff', '12628780_15.tiff', '11278930_15.tiff', '18628900_15.tiff', '26129290_15.tiff', '20129005_15.tiff', '17278960_15.tiff', '24778915_15.tiff', '19528630_15.tiff', '23428570_15.tiff', '17578720_15.tiff', '22829065_15.tiff', '18028870_15.tiff', '16228930_15.tiff', '21329005_15.tiff', '23429170_15.tiff', '18028705_15.tiff', '16078960_15.tiff', '17728990_15.tiff', '10828900_15.tiff', '10228780_15.tiff', '10678795_15.tiff', '21479050_15.tiff', '22529485_15.tiff', '18178705_15.tiff', '17428855_15.tiff', '26728675_15.tiff', '26128705_15.tiff', '18628885_15.tiff', '10828825_15.tiff', '24328885_15.tiff', '11428930_15.tiff', '17278780_15.tiff', '18928795_15.tiff']
def read_Mas_datasets(root_path, mode='train'):
    images = []
    masks = []
    image_root = os.path.join(root_path, 'train/images/')
    gt_root = os.path.join(root_path, 'train/masks_road/')
    for image_name in os.listdir(image_root):
        # 过滤掉超过10%的空白图片
        if image_name in removeImage:
            continue
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.tiff')
        label_path = os.path.join(gt_root, image_name.split('.')[0] + '.tif')

        # image_path = os.path.join(image_root, image_name)
        # label_path = os.path.join(gt_root, image_name)

        images.append(image_path)
        masks.append(label_path)

    # ####数量太多，暂不输出显示:
    # print(images, masks)
    return images, masks


def read_CHG_datasets(root_path, mode='train'):
    images = []
    masks = []

    image_root = os.path.join(root_path, 'train/images/')
    gt_root = os.path.join(root_path, 'train/masks1/')
    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.jpg')
        label_path = os.path.join(gt_root, image_name.split('_')[0] + '_mask.jpg')

        # image_path = os.path.join(image_root, image_name)
        # label_path = os.path.join(gt_root, image_name)

        images.append(image_path)
        masks.append(label_path)

    # ####数量太多，暂不输出显示:
    # print(images, masks)
    return images, masks


def read_ORIGA_datasets(root_path, mode='train'):
    images = []
    masks = []

    if mode == 'train':
        read_files = os.path.join(root_path, 'Set_A.txt')
    else:
        read_files = os.path.join(root_path, 'Set_B.txt')

    image_root = os.path.join(root_path, 'images')
    gt_root = os.path.join(root_path, 'masks')

    for image_name in open(read_files):
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.jpg')
        label_path = os.path.join(gt_root, image_name.split('.')[0] + '.jpg')

        print(image_path, label_path)

        images.append(image_path)
        masks.append(label_path)

    return images, masks


def read_Messidor_datasets(root_path, mode='train'):
    images = []
    masks = []

    if mode == 'train':
        read_files = os.path.join(root_path, 'train.txt')
    else:
        read_files = os.path.join(root_path, 'test.txt')

    image_root = os.path.join(root_path, 'save_image')
    gt_root = os.path.join(root_path, 'save_mask')

    for image_name in open(read_files):
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.jpg')
        label_path = os.path.join(gt_root, image_name.split('_')[0] + '_mask.png')

        images.append(image_path)
        masks.append(label_path)

    return images, masks


def read_RIM_ONE_datasets(root_path, mode='train'):
    images = []
    masks = []

    if mode == 'train':
        read_files = os.path.join(root_path, 'train_files.txt')
    else:
        read_files = os.path.join(root_path, 'test_files.txt')

    image_root = os.path.join(root_path, 'RIM-ONE-images')
    gt_root = os.path.join(root_path, 'RIM-ONE-exp1')

    for image_name in open(read_files):
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.png')
        label_path = os.path.join(gt_root, image_name.split('.')[0] + '-exp1.png')

        images.append(image_path)
        masks.append(label_path)

    return images, masks


def read_DRIVE_datasets(root_path, mode='train'):
    images = []
    masks = []

    image_root = os.path.join(root_path, 'train/images/')
    gt_root = os.path.join(root_path, 'train/masks/')

    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.jpg')
        # print(image_path)
        label_path = os.path.join(gt_root, image_name.split('_')[0] + '_mask.png')
        # print(label_path)

        # image_path = os.path.join(image_root, image_name)
        # label_path = os.path.join(gt_root, image_name)

        images.append(image_path)
        masks.append(label_path)

    # ####数量太多，暂不输出显示:
    # print(images, masks)
    return images, masks


def read_Cell_datasets(root_path, mode='train'):
    images = []
    masks = []

    image_root = os.path.join(root_path, 'train-images')
    gt_root = os.path.join(root_path, 'train-labels')

    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name)
        label_path = os.path.join(gt_root, image_name)

        images.append(image_path)
        masks.append(label_path)

    print(images, masks)

    return images, masks


def read_datasets_vessel(root_path, mode='train'):
    images = []
    masks = []

    image_root = os.path.join(root_path, 'training/images')
    gt_root = os.path.join(root_path, 'training/mask')

    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name)
        label_path = os.path.join(gt_root, image_name)

        if cv2.imread(image_path) is not None:

            if os.path.exists(image_path) and os.path.exists(label_path):
                images.append(image_path)
                masks.append(label_path)

    print(images[:10], masks[:10])

    return images, masks


class ImageFolder(data.Dataset):

    def __init__(self, root_path, datasets='Messidor', mode='train'):
        self.root = root_path
        self.mode = mode
        self.dataset = datasets
        assert self.dataset in ['RIM-ONE', 'Messidor', 'ORIGA', 'DRIVE', 'Cell', 'Vessel', 'Mas', 'CHG'], \
            "the dataset should be in 'Messidor', 'ORIGA', 'RIM-ONE', 'Vessel' "
        if self.dataset == 'RIM-ONE':
            self.images, self.labels = read_RIM_ONE_datasets(self.root, self.mode)
        elif self.dataset == 'Messidor':
            self.images, self.labels = read_Messidor_datasets(self.root, self.mode)
        elif self.dataset == 'ORIGA':
            self.images, self.labels = read_ORIGA_datasets(self.root, self.mode)
        elif self.dataset == 'DRIVE':
            self.images, self.labels = read_DRIVE_datasets(self.root, self.mode)
        elif self.dataset == 'Cell':
            self.images, self.labels = read_Cell_datasets(self.root, self.mode)
        elif self.dataset == 'GAN_Vessel':
            self.images, self.labels = read_datasets_vessel(self.root, self.mode)
        elif self.dataset == 'Mas':
            self.images, self.labels = read_Mas_datasets(self.root, self.mode)
        elif self.dataset == 'CHG':
            self.images, self.labels = read_CHG_datasets(self.root, self.mode)
        else:
            print('Default dataset is Messidor')
            self.images, self.labels = read_Messidor_datasets(self.root, self.mode)

    def __getitem__(self, index):

        img, mask = default_DRIVE_loader(self.images[index], self.labels[index])
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)
