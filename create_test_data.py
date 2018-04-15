import random
import _pickle
import math

def random_rgb():
    ret_col = []
    for _ in range(3):
        ret_col.append(random.randint(0, 255))
    return ret_col


def save_list(s_list, name):
    print("saving...")
    with open(name, 'wb') as fp:
        _pickle.dump(s_list, fp)


def white_or_black(rgb):
    rgb_new = [0.0, 0.0, 0.0]
    for color_index in range(len(rgb)):
        s_c = rgb[color_index] / 255.0
        if s_c <= 0.03928:
            s_c = s_c / 12.92
        else:
            s_c = math.pow(((s_c+0.055) / 1.055), 2.4)
        rgb_new[color_index] = s_c

    L = 0.2126 * rgb_new[0] + 0.7152 * rgb_new[1] + 0.0722 * rgb_new[2]

    if L > 0.179:
        # black text
        return [0, 1]
    else:
        # white text
        return [1, 0]


def scale_rgb(rgb):
    new_rgb = [0.0] * 3
    for i in range(3):
        new_rgb[i] = rgb[i]/255
    return new_rgb


if __name__ == "__main__":
    amount_training_data = 40000
    data_list_training = []
    for _ in range(amount_training_data):
        rgb = random_rgb()
        w_b = white_or_black(rgb)
        data_list_training.append((scale_rgb(rgb), w_b))

    amount_validation_data = 2000
    data_list_validation = []
    for _ in range(amount_validation_data):
        rgb = random_rgb()
        w_b = white_or_black(rgb)
        data_list_validation.append((scale_rgb(rgb), w_b))

    save_list(data_list_training, "training_data")
    save_list(data_list_validation, "validation_data")
