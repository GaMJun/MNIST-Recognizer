import cv2
import glob
import numpy as np


def read_all_images():
    print("\nReading all images on folder MNIST_Dataset\n")
    size = 0
    image_list = []
    for number_index in range(10):
        print("Number of ", number_index, "- ", end='')
        image_list.append([])
        for img_filename in glob.glob('./MNIST_Dataset/' + str(number_index) + '/*.jpg'):
            img_temp = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)
            (thresh, bw_img) = cv2.threshold(img_temp, 25, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            image_list[number_index].append(bw_img)
            size = size + 1
        print(len(image_list[number_index]))

    print("Total number of images -", size)
    return image_list


def block_shaper(arr, n_rows, n_cols):
    h, w = arr.shape
    assert h % n_rows == 0, "{} rows is not evenly divisble by {}".format(h, n_cols)
    assert w % n_cols == 0, "{} cols is not evenly divisble by {}".format(w, n_cols)
    return (arr.reshape(h // n_rows, n_rows, -1, n_cols)
            .swapaxes(1, 2)
            .reshape(-1, n_rows, n_cols))


def first_feature_extraction(image_list):
    print("Extracting features by the first approach")
    result = []
    for number_index in range(10):
        for img_index in range(len(image_list[number_index])):
            blocks = block_shaper(image_list[number_index][img_index], 7, 7)
            ratio_block = []
            for index_block in range(len(blocks)):
                ratio_block.append(len(np.argwhere(blocks[index_block] == 255)) / 49)
            ratio_block.append(number_index)
            result.append(ratio_block)

    with open("./Extracted_Features/first_features.csv", "w") as fp:
        for row in range(len(result)):
            fp.write(
                "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d\n" % (
                    result[row][1], result[row][2], result[row][4], result[row][5], result[row][6],
                    result[row][7],
                    result[row][8], result[row][9], result[row][10], result[row][11], result[row][13],
                    result[row][14], result[row][16]))
    print("done")


def second_feature_extraction(image_list):
    print("Extracting features by the second approach")
    result = []
    for number_index in range(10):
        for img_index in range(len(image_list[number_index])):
            size = np.size(image_list[number_index][img_index])
            skeleton = np.zeros(image_list[number_index][img_index].shape, np.uint8)

            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            done = False
            # cv2.imshow("original", image_list[number_index][img_index])
            while not done:

                eroded = cv2.erode(image_list[number_index][img_index], element)
                temp = cv2.dilate(eroded, element)
                temp = cv2.subtract(image_list[number_index][img_index], temp)
                skeleton = cv2.bitwise_or(skeleton, temp)
                image_list[number_index][img_index] = eroded.copy()

                zeros = size - cv2.countNonZero(image_list[number_index][img_index])
                if zeros == size:
                    done = True

            # cv2.imshow("skeleton", skeleton)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            blocks = block_shaper(skeleton, 7, 7)
            ratio_block = []
            for index_block in range(len(blocks)):
                ratio_block.append(len(np.argwhere(blocks[index_block] == 255)) / 49)
            ratio_block.append(number_index)
            result.append(ratio_block)

    with open("./Extracted_Features/second_features.csv", "w") as fp:
        for row in range(len(result)):
            fp.write(
                "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d\n" % (
                    result[row][1], result[row][2], result[row][4], result[row][5], result[row][6],
                    result[row][7],
                    result[row][8], result[row][9], result[row][10], result[row][11], result[row][13],
                    result[row][14], result[row][16]))
    print("done")
