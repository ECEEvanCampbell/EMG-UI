import pkgutil
import pickle
import matplotlib.pyplot as plt

def read_pickle(location):
    with open(location, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    data = read_pickle('results/FLT_SXXX_MXXX_TXXX.pkl')
    unzipped_object = zip(*data['cursor_position'])
    unzipped_list   = list(unzipped_object)
    # unzipped_emg = zip(*data['EMG'])
    # unzipped_list_emg = list(unzipped_emg)
    # x = len(unzipped_list_emg)
    # emg = data['EMG']
    # extracted_emg = emg[20]
    # ch_5 = extracted_emg[7]
    # x = len(ch_5)

    # Plot the path taken and the emg data as subplots
    fig, ax = plt.subplots(2)

    ax[0].scatter(unzipped_list[0], unzipped_list[1])
    # plt.xlim((0, 1250))
    # plt.ylim((0,750))
    # plt.show()
    # A = 1

    # ax[1].plot(x, ch_5)
    plt.show()
