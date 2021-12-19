# --------------------------------------------------------
# P2ORM: Formulation, Inference & Application
# Licensed under The MIT License [see LICENSE for details]
# Written by Xuchong Qiu
# --------------------------------------------------------
# image processing utility


from PIL import Image, ImageMath
import numpy as np


def paste_with_mask(im_fg, im_bkg, mask_fg):
    """
    paste fg on bg with binary mask
    :param im_fg: PIL
    :param im_bkg: PIL
    :param mask_fg: numpy [0,1]
    :return: PIL
    """
    im_comp = im_bkg.copy()
    mask_fg = Image.fromarray(np.uint8(mask_fg * 255))  # pil mask
    im_comp.paste(im_fg, (0, 0), mask_fg)  # paste fg according to mask

    return im_comp


def paste_mat_with_mask(im_fg, im_bkg, mask_fg, mat_aplpha=0.5):
    """
    paste fg on bg with binary mask and do blending to make fg transparent
    :param im_fg: PIL
    :param im_bkg: PIL
    :param mask_fg: numpy [0,1]
    :param mat_aplpha:
    :return: PIL
    """
    im_comp = im_bkg.copy()
    mask_fg = Image.fromarray(np.uint8(mask_fg * 255))  # pil mask
    im_comp.paste(im_fg, (0, 0), mask_fg)  # paste fg according to mask
    im_comp_mat = Image.blend(im_bkg, im_comp, alpha=mat_aplpha)  # matting

    return im_comp_mat


def resize_padding(im_ori, size_out, obj_box_in):
    """resize to render img size with fix aspect ratio and padding if needed"""
    ori_sz = np.array(im_ori.size)  # width, height
    rescales = size_out / ori_sz
    zoom_factor = 0.75  # for conserving black border
    scale_ratio = min(rescales) * zoom_factor
    im_resz_sz = (ori_sz * scale_ratio).astype('int')
    im_resz = im_ori.resize(im_resz_sz, Image.ANTIALIAS)

    obj_box_in_sz  = [obj_box_in[2] - obj_box_in[0], obj_box_in[3] - obj_box_in[1]]
    obj_box_new_sz = np.array(obj_box_in_sz) * scale_ratio

    # create a new image and paste the resized one on it
    im_new = Image.new("RGB", (size_out[0], size_out[1]))
    im_new.paste(im_resz, ((size_out[0] - im_resz_sz[0]) // 2,
                           (size_out[1] - im_resz_sz[1]) // 2))

    return im_new, scale_ratio, obj_box_new_sz


def center_to_gt(bgr_gl, depth_gl):
    """make object rendered/depth bbox centered in image"""

    im_rendered     = Image.fromarray(bgr_gl, "RGB")
    bbox_rendered   = im_rendered.getbbox()
    box_sz_rendered = np.array([bbox_rendered[2] - bbox_rendered[0],
                                bbox_rendered[3] - bbox_rendered[1]])

    # move rendered obj bbox to image center
    im_ren_crop = im_rendered.crop(bbox_rendered)
    im_ren_new  = Image.new("RGB", (im_rendered.size[0], im_rendered.size[1]))
    im_ren_new.paste(im_ren_crop, ((im_rendered.size[0] - box_sz_rendered[0]) // 2,
                                   (im_rendered.size[1] - box_sz_rendered[1]) // 2))

    # center depth map too
    depth_rendered = Image.fromarray(depth_gl / depth_gl.max() * 255)
    depth_ren_crop = depth_rendered.crop(bbox_rendered)
    depth_ren_new  = Image.new("P", (im_rendered.size[0], im_rendered.size[1]))
    depth_ren_new.paste(depth_ren_crop, ((im_rendered.size[0] - box_sz_rendered[0]) // 2,
                                         (im_rendered.size[1] - box_sz_rendered[1]) // 2))

    return im_ren_new, depth_ren_new


def square_crop(im_in, bbox_in, extend_ratio):
    """crop a square region by enlarging box_in(largest dim) with extend ratio"""
    bbox_in_sz = [bbox_in[2] - bbox_in[0], bbox_in[3] - bbox_in[1]]
    center_in  = [bbox_in[0] + bbox_in_sz[0] / 2, bbox_in[1] + bbox_in_sz[1] / 2]

    bbox_new_len = extend_ratio * np.array(max(bbox_in_sz))
    bbox_new_sz  = np.array([bbox_new_len, bbox_new_len])
    bbox_new = [center_in[0] - bbox_new_sz[0] / 2, center_in[1] - bbox_new_sz[1] / 2,
                center_in[0] + bbox_new_sz[0] / 2, center_in[1] + bbox_new_sz[1] / 2]
    im_crop = im_in.crop(bbox_new)

    return im_crop, bbox_new


def change_background(img, mask, bg):
    ow, oh = img.size
    bg = bg.resize((ow, oh)).convert('RGB')

    imcs = list(img.split())
    bgcs = list(bg.split())
    maskcs = list(mask.split())
    fics = list(Image.new(img.mode, img.size).split())

    for c in range(len(imcs)):
        negmask = maskcs[c].point(lambda i: 1 - i / 255)
        posmask = maskcs[c].point(lambda i: i / 255)
        fics[c] = ImageMath.eval("a * c + b * d", a=imcs[c], b=bgcs[c], c=posmask, d=negmask).convert('L')
    out = Image.merge(img.mode, tuple(fics))

    return out