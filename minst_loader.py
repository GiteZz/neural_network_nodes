from mnist import MNIST
import time
# converts value to vector: 0 -> [1,0,0,...,0]
def to_vector(value):
    ret_list = [0] * 10
    ret_list[value] = 1
    return ret_list

def get_mnist_data():
    mndata = MNIST('D:\\Programmeren\\Python\\neural_network_nodes\\mnist')
    mndata.gz = True
    training = mndata.load_training()
    training_img = training[0]
    training_res = training[1]
    training_tuple = []

    testing = mndata.load_testing()
    testing_img = testing[0]
    testing_res = testing[1]
    testing_tuple = []

    for i in range(len(testing_res)):
        testing_tuple.append((testing_img[i], to_vector(testing_res[i])))

    for i in range(len(training_res)):
        training_tuple.append((training_img[i], to_vector(training_res[i])))

    return training_tuple, testing_tuple


if __name__ == "__main__":
    start_time = time.time()

    training_t, test_t = get_mnist_data()
    print("--- %s seconds ---" % (time.time() - start_time))
    a = 5