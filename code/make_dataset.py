import numpy as np
import cv2
import random
import torch
import os
from torch.utils.data import Dataset
from tqdm import tqdm

# script to generate a dataset of images of a claw game from top down view
seed = 438
np.random.seed(seed)
torch.random.manual_seed(seed)
random.seed(seed)

# inputs
DATASET_PATH_CLAW = "dataset"
CLAW_ASSETS_PATH = "claw_assets"
SAVE_FILE_PATH = "figures"
N_SAMPLES = 20000

IMAGE_PATH_ALL = [
    "teddy_view1.png",
    "teddy_view2.png",
    "teddy_view3.png",
    "teddy_view4.png",
    "ball1.png",
    "ball2.png",
    "ball3.png",
    "ball4.png",
]

os.makedirs(DATASET_PATH_CLAW, exist_ok=True)
os.makedirs(SAVE_FILE_PATH, exist_ok=True)


class ClawCustomDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        # just load it all into RAM
        self.image_all = np.load(
            os.path.join(dataset_path, "images.npy"), allow_pickle=True
        )
        self.label_all = np.load(
            os.path.join(dataset_path, "labels.npy"), allow_pickle=True
        )
        self.action_all = np.load(
            os.path.join(dataset_path, "actions.npy"), allow_pickle=True
        )
        self.transform = transform

    def __len__(self):
        return self.image_all.shape[0]

    def __getitem__(self, index):
        image = self.image_all[index]
        label = self.label_all[index]
        action = self.action_all[index]
        if self.transform:
            image = self.transform(image)
        return (image, label, action)


def scale_image(img, scale_factor):
    # scale_factor: fraction of original size
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img


def rotate_image(img, angle):
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return img


def get_random_offset(canvas, img):
    # choose a random location
    img_width = img.shape[1]
    img_height = img.shape[0]
    canvas_width = canvas.shape[1]
    canvas_height = canvas.shape[0]

    rand_xpos = np.random.randint(0, canvas_width - img_width)
    rand_ypos = np.random.randint(0, canvas_height - img_height)

    return rand_xpos, rand_ypos


def put_foreground_on_background(
    background, foreground, obj_mask, x_offset=None, y_offset=None
):
    # from https://stackoverflow.com/questions/40895785/using-opencv-to-overlay-transparent-image-onto-another-image
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert (
        bg_channels == 3
    ), f"background image should have exactly 3 channels (RGB). found:{bg_channels}"
    assert (
        fg_channels == 4
    ), f"foreground image should have exactly 4 channels (RGBA). found:{fg_channels}"

    # center by default
    if x_offset is None:
        x_offset = (bg_w - fg_w) // 2
    if y_offset is None:
        y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1:
        return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y : fg_y + h, fg_x : fg_x + w]
    background_subsection = background[bg_y : bg_y + h, bg_x : bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))
    composite = (
        background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask
    )
    obj_mask_subsection = alpha_channel[:, :, None] > 0.5

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite
    obj_mask[bg_y:bg_y + h, bg_x:bg_x + w] += obj_mask_subsection * 1
    return background, obj_mask


def increase_brightness(img, value=30):
    # need to modify this to do alpha channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def get_datapoint(x_size, img_setup_dict):
    # set up canvas
    # canvas = np.zeros((x_size[0],x_size[1],3), dtype=np.uint8) # BGR
    canvas = cv2.imread(os.path.join(CLAW_ASSETS_PATH, "bg3.png"), cv2.IMREAD_UNCHANGED)
    canvas = cv2.resize(canvas, x_size, interpolation=cv2.INTER_AREA)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)

    obj_mask = np.zeros(
        (x_size[0], x_size[1], 1), dtype=float
    )  # keep track of where overlaid images

    img_idx = np.random.randint(0, len(img_setup_dict))
    # img_idx = 8
    img_filenames, scales, angles, x_offsets, y_offsets = img_setup_dict[img_idx]
    for i in range(len(img_filenames)):
        image_path = os.path.join(CLAW_ASSETS_PATH, img_filenames[i])
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # if 'teddy' in image_path:
        #     img = increase_brightness(img, 400)
        img = scale_image(img, scales[i])
        img = rotate_image(img, angles[i])
        canvas, obj_mask = put_foreground_on_background(
            canvas, img, obj_mask, x_offsets[i], y_offsets[i]
        )

    # randomly pick on index where obj_mask>0
    # for actions, randomly pick a point on one of our objects
    # (any better way to do this?)
    obj_mask_flat = obj_mask.flatten()
    arr_idx0 = np.zeros_like(obj_mask, dtype=np.uint8)
    arr_idx1 = np.zeros_like(obj_mask, dtype=np.uint8)
    for i in range(obj_mask.shape[0]):
        arr_idx0[i, :] = i
        arr_idx1[:, i] = i
    arr_idx0 = arr_idx0.flatten()
    arr_idx1 = arr_idx1.flatten()
    # idx_possible = obj_mask_flat>0.001
    idx_possible = obj_mask_flat > 0.5
    idx_choose = np.random.randint(0, idx_possible.sum())
    # obj_mask_flat[idx_possible][idx_choose]
    idx = (arr_idx0[idx_possible][idx_choose], arr_idx1[idx_possible][idx_choose])
    target = np.zeros_like(obj_mask)
    target[idx[0], idx[1]] = 1
    action = idx  # just x and y coords of image
    action = (idx[1], idx[0])  # WARNING we have to swap these over

    # convert to rgb
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # create a smaller 32x32 version for quick training
    canvas_small = cv2.resize(canvas, (32, 32))

    return canvas, canvas_small, action, obj_mask, target


def main():
    x_size = (64, 64)

    scale_ball = 0.09
    scale_bear = 0.3
    img_setup_dict = {}
    img_setup_dict[0] = (["ball1.png"], [scale_ball], [5], [12], [10])
    # img_setup_dict[1] = (['teddy_view3.png'], [scale_bear], [10], [20], [15])
    img_setup_dict[2] = (
        ["ball1.png", "ball4.png"],
        [scale_ball, scale_ball],
        [5, 90],
        [4, 40],
        [10, 30],
    )
    # img_setup_dict[3] = (['ball1.png', 'teddy_view3.png'], [scale_ball, scale_bear], [34, 180], [4, 35], [12, 30])
    # img_setup_dict[4] = (['teddy_view3.png', 'teddy_view2.png'], [scale_bear*1.1, scale_bear], [34, 0], [1, 38], [12, 30])
    img_setup_dict[5] = (
        ["ball1.png", "ball4.png", "ball3.png"],
        [scale_ball, scale_ball, scale_ball],
        [5, 90, 0],
        [4, 40, 6],
        [10, 30, 35],
    )

    img_setup_dict[6] = (["snake_view1.png"], [scale_bear * 0.9], [10], [15], [15])
    img_setup_dict[1] = (["pikachu_view1.png"], [scale_bear * 0.8], [0], [5], [15])
    img_setup_dict[3] = (
        ["ball3.png", "snake_view1.png"],
        [scale_ball, scale_bear * 0.8],
        [34, 3],
        [4, 25],
        [12, 5],
    )
    # img_setup_dict[4] = (['pikachu_view1.png', 'teddy_view3.png'], [scale_bear*0.8, scale_bear], [10, 3], [10, 0], [15, 0])
    # img_setup_dict[8] = (['pikachu_view1.png', 'teddy_view3.png', 'ball4.png'], [scale_bear*0.8, scale_bear, scale_ball*0.9], [10, 3, 180], [10, 0, 40], [15, 0, 5])
    img_setup_dict[4] = (
        ["pikachu_view1.png", "ball4.png"],
        [scale_bear * 0.8, scale_ball * 0.9],
        [10, 180],
        [10, 40],
        [15, 5],
    )
    # img_setup_dict[8] = (['panda_view1.png'], [scale_bear], [20], [15], [15])

    canvas, canvas_small, action, obj_mask, target = get_datapoint(
        x_size, img_setup_dict
    )

    x_all = []
    x_small_all = []
    action_all = []
    obj_mask_all = []
    for i in tqdm(range(N_SAMPLES), desc="datapoint"):
        canvas, canvas_small, action, obj_mask, target = get_datapoint(
            x_size, img_setup_dict
        )
        x_all.append(canvas)
        x_small_all.append(canvas_small)
        action_all.append(action)
        obj_mask_all.append(obj_mask)

    x_all = np.array(x_all)
    x_small_all = np.array(x_small_all)
    action_all = np.array(action_all)
    obj_mask_all = np.array(obj_mask_all)

    # save as new dataset
    np.save(os.path.join(DATASET_PATH_CLAW, "images"), x_all, allow_pickle=True)
    np.save(
        os.path.join(DATASET_PATH_CLAW, "images_small"), x_small_all, allow_pickle=True
    )
    np.save(os.path.join(DATASET_PATH_CLAW, "labels"), obj_mask_all, allow_pickle=True)
    np.save(os.path.join(DATASET_PATH_CLAW, "actions"), action_all, allow_pickle=True)


if __name__ == "__main__":
    main()
