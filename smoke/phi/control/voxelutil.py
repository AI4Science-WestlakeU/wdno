from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
from phi.fluidformat import *


def text_to_pixels(text, size=10, binary=False, as_numpy_array=True):
    image = Image.new("1" if binary else "L", (len(text)*size*3//4, size), 0)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except:
        font = ImageFont.truetype('Pillow/Tests/fonts/DejaVuSans.ttf', size=size)
    draw.text((0,0), text, fill=255, font=font)
    del draw

    if as_numpy_array:
        return np.array(image).astype(np.float32) / 255.0
    else:
        return image


# image = text_to_pixels("The", as_numpy_array=False)
# image.save("testimg.png", "PNG")


def alphabet_soup(shape, count, margin=1, total_content=100, fontsize=10):
    if len(shape) != 4: raise ValueError("shape must be 4D")
    array = np.zeros(shape)
    letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    for batch in range(shape[0]):
        for i in range(count):
            letter = letters[random.randint(0, len(letters)-1)]
            tile = text_to_pixels(letter, fontsize)#[::-1, :]
            y = random.randint(margin, shape[1] - margin - tile.shape[0] - 2)
            x = random.randint(margin, shape[2] - margin - tile.shape[1] - 2)
            array[batch, y:(y+tile.shape[0]), x:(x+tile.shape[1]), 0] += tile

    return array.astype(np.float32) * total_content / np.sum(array)


def random_word(shape, min_count, max_count, margin=1, total_content=100, fontsize=10, y=40):
    if len(shape) != 4: raise ValueError("shape must be 4D")
    array = np.zeros(shape)
    letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    for b in range(shape[0]):
        count = random.randint(min_count, max_count)
        for i in range(count):
            letter = letters[random.randint(0, len(letters)-1)]
            tile = text_to_pixels(letter, fontsize)#[::-1, :]
            x = random.randint(margin, shape[2] - margin - tile.shape[1] - 2)
            array[b, y:(y+tile.shape[0]), x:(x+tile.shape[1]), 0] += tile

    return array.astype(np.float32) * total_content / np.sum(array)



def single_shape(shape, scene, margin=1, fluid_mask=None):
    if len(shape) != 4: raise ValueError("shape must be 4D")
    array = np.zeros(shape)
    for batch in range(shape[0]):
        img = scene.read_array("Shape", random.choice(scene.indices))[0,...]
        while True:
            y = random.randint(margin, shape[1] - margin - img.shape[0] - 2)
            x = random.randint(margin, shape[2] - margin - img.shape[1] - 2)
            array[batch, y:(y + img.shape[0]), x:(x + img.shape[1]), :] = img
            if _all_density_valid(array[batch:batch+1,...], fluid_mask):
                break
            else:
                array[batch,...] = 0

    return array.astype(np.float32)


def _all_density_valid(density, fluid_mask):
    if fluid_mask is None:
        return True
    return np.sum(density * fluid_mask) == np.sum(density)


def push_density_inside(density_tile, tile_location, fluid_mask): # (y, x)
    """
Tries to adjust the tile_location so that the density_tile does not overlap with any obstacles.
    :param density_tile: 2D binary array, representing the density mask to be shifted
    :param tile_location: the initial location of the tile, (1D array with 2 values)
    :param fluid_mask: 2D binary array (must be larger than the tile)
    :return: the shifted location (1D array with 2 values)
    """
    x, y = np.meshgrid(*[np.linspace(-1, 1, d) for d in density_tile.shape])
    location = np.array(tile_location, dtype=np.int)

    def cropped_mask(location):
        slices = [slice(location[i], location[i]+density_tile.shape[i]) for i in range(2)]
        return fluid_mask[slices]

    while True:
        cropped_fluid_mask = cropped_mask(location)
        overlap = density_tile * (1-cropped_fluid_mask)
        if np.sum(overlap) == 0:
            return location
        update = -np.sign([np.sum(overlap * y), np.sum(overlap * x)]).astype(np.int)
        if np.all(update == 0):
            raise ValueError("Failed to push tile with initial location %s out of obstacle" % (tile_location,))
        location += update


# print(alphabet_soup([1, 16, 16, 1], 1000)[0,:,:,0])

# result = single_shape((2, 64, 64, 1), scene_at("data/shapelib/sim_000000"))
# print(result.shape, np.sum(result))

# Test push_density_inside
# fluid_mask = np.ones([64, 64])
# fluid_mask[10:20, 10:20] = 0
# density_tile = np.ones([5,5])
# tile_location = (18,9)
# print(push_density_inside(density_tile, tile_location, fluid_mask))