import os

import argparse
import shutil
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-d', '--dataset', help='Path to dataset', required=True)

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print("File not found: {}".format(args.dataset))
        exit(1)

    # print all files that are in images and not in labels
    images = os.listdir(os.path.join(args.dataset, 'images/train'))
    labels = os.listdir(os.path.join(args.dataset, 'labels_gs/train'))

    # make trash dir
    os.makedirs(os.path.join(args.dataset, 'trash'), exist_ok=True)

    for image in images:
        label = image.replace('.jpg', '_bin.png')
        if label not in labels:
            print("Missing label: {}".format(label))
            # Move image to trash
            try:
                shutil.move(os.path.join(args.dataset, 'images/train', image),
                            os.path.join(args.dataset, 'trash', image))
                print("Moved image: {} to trash".format(image))
            except FileNotFoundError:
                print("File not found: {}".format(image))

    # print all files that are in labels and not in images
    for label in labels:
        image = label.replace('_bin.png', '.jpg')
        if image not in images:
            print("Missing image: {}".format(image))
            # Move label to trash
            try:
                shutil.move(os.path.join(args.dataset, 'labels_gs/train', label),
                            os.path.join(args.dataset, 'trash', label))
                print("Moved label: {} to trash".format(label))
            except FileNotFoundError:
                print("File not found: {}".format(label))