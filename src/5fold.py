import pickle
import os


def get_image_list():
    pre_path = "../Data/twitter/"
    image_train = [image.split('.')[0] for image in os.listdir(pre_path + 'images_train')]
    image_text = [image.split('.')[0] for image in os.listdir(pre_path + 'images_validation')]
    images = list(image_train) + list(image_text)
    return images


def get_idx(images):
    pre_path = "../Data/twitter/"
    texts = open(pre_path + 'train_posts.txt', 'r').readlines()
    train_ids = []
    for i, line in enumerate(texts):
        if i == 0: continue
        line_data = []
        image_id = line.split('\t')[3].lower().split(',')
        continue_flag = False
        for image in image_id:
            if image in images:
                image_id = image
                continue_flag = True
                break
        if not continue_flag:
            continue
        # post_id = line.split('\t')[0]
        train_ids.append(line.split('\t'))
    pickle.dump(train_ids, open(pre_path + 'train_id.pkl', 'wb'))


def get5fold(dataset):
    data_paths = ["../Data/" + str(dataset) + "/train_id.pickle", "../Data/" + str(dataset) + "/validate_id.pickle"]
    # , "../Data/" + str(dataset) + "/test_id.pickle"]
    all_data = {}
    for path in data_paths:
        data = pickle.load(open(path, 'rb'))
        all_data.update(data)
    fold_num = 5
    num_per_fold = int(len(all_data) / fold_num) + 1
    all_data_list = list(all_data.items())
    fold1 = dict(all_data_list[0:num_per_fold])
    fold2 = dict(all_data_list[num_per_fold: num_per_fold * 2])
    fold3 = dict(all_data_list[num_per_fold * 2: num_per_fold * 3])
    fold4 = dict(all_data_list[num_per_fold * 3: num_per_fold * 4])
    fold5 = dict(all_data_list[num_per_fold * 4: len(all_data)])
    train_id1 = {}
    train_id2 = {}
    train_id3 = {}
    train_id4 = {}
    train_id5 = {}
    test_id1 = {}
    test_id2 = {}
    test_id3 = {}
    test_id4 = {}
    test_id5 = {}

    train_id1.update(fold1)
    train_id1.update(fold2)
    train_id1.update(fold3)
    train_id1.update(fold4)
    test_id1.update(fold5)

    train_id2.update(fold2)
    train_id2.update(fold3)
    train_id2.update(fold4)
    train_id2.update(fold5)
    test_id2.update(fold1)

    train_id3.update(fold1)
    train_id3.update(fold3)
    train_id3.update(fold4)
    train_id3.update(fold5)
    test_id3.update(fold2)

    train_id4.update(fold1)
    train_id4.update(fold2)
    train_id4.update(fold4)
    train_id4.update(fold5)
    test_id4.update(fold3)

    train_id5.update(fold1)
    train_id5.update(fold2)
    train_id5.update(fold3)
    train_id5.update(fold5)
    test_id5.update(fold4)

    pickle.dump(train_id1, open('../Data/' + str(dataset) + '/train_id1.pkl', 'wb'))
    pickle.dump(test_id1, open('../Data/' + str(dataset) + '/test_id1.pkl', 'wb'))
    print("Train dataset1 length: ", len(train_id1), "Test dataset1 length: ", len(test_id1))

    pickle.dump(train_id2, open('../Data/' + str(dataset) + '/train_id2.pkl', 'wb'))
    pickle.dump(test_id2, open('../Data/' + str(dataset) + '/test_id2.pkl', 'wb'))
    print("Train dataset2 length: ", len(train_id2), "Test dataset2 length: ", len(test_id2))

    pickle.dump(train_id3, open('../Data/' + str(dataset) + '/train_id3.pkl', 'wb'))
    pickle.dump(test_id3, open('../Data/' + str(dataset) + '/test_id3.pkl', 'wb'))
    print("Train dataset3 length: ", len(train_id3), "Test dataset1 length: ", len(test_id3))

    pickle.dump(train_id4, open('../Data/' + str(dataset) + '/train_id4.pkl', 'wb'))
    pickle.dump(test_id4, open('../Data/' + str(dataset) + '/test_id4.pkl', 'wb'))
    print("Train dataset4 length: ", len(train_id4), "Test dataset1 length: ", len(test_id4))

    pickle.dump(train_id5, open('../Data/' + str(dataset) + '/train_id5.pkl', 'wb'))
    pickle.dump(test_id5, open('../Data/' + str(dataset) + '/test_id5.pkl', 'wb'))
    print("Train dataset5 length: ", len(train_id5), "Test dataset5 length: ", len(test_id5))


def get5foldTwitter():
    data_path = "../Data/twitter/train_id.pkl"
    data = pickle.load(open(data_path, 'rb'))
    fold_num = 5
    num_per_fold = int(len(data) / fold_num) + 1
    fold1 = data[0:num_per_fold]
    fold2 = data[num_per_fold: num_per_fold * 2]
    fold3 = data[num_per_fold * 2: num_per_fold * 3]
    fold4 = data[num_per_fold * 3: num_per_fold * 4]
    fold5 = data[num_per_fold * 4: len(data)]
    train_id1 = fold1 + fold2 + fold3 + fold4
    test_id1 = fold5

    train_id2 = fold1 + fold2 + fold3 + fold4
    test_id2 = fold5

    train_id3 = fold1 + fold2 + fold3 + fold4
    test_id3 = fold5

    train_id4 = fold1 + fold2 + fold3 + fold4
    test_id4 = fold5

    train_id5 = fold1 + fold2 + fold3 + fold4
    test_id5 = fold5


if __name__ == '__main__':
    image_list = get_image_list()
    get_idx(image_list)
    get5fold('twitter')
