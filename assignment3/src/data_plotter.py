import matplotlib.pyplot as plt
from ast import literal_eval as make_tuple

def plot_errors(errors, title=''):
    tr_l = []
    te_l = []
    tr_a = []
    te_a = []
    x_axis = []
    count = 0
    for error in errors:
        count += 1
        tr_l.append(error[0])
        te_l.append(error[1])
        tr_a.append(error[2])
        te_a.append(error[3])
        x_axis.append(count)

    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.xlabel('Half Epoch')
    plt.ylabel('Squared Loss Error')
    plt.title('Hidden Nodes = %s' % title)
    plt.plot(x_axis, tr_l, 'g^', x_axis, te_l, 'bs')
    #plt.show()

    epoch_count = len(tr_a)
    two_thirds = int(epoch_count * 2 / 3)
    for i in range(epoch_count):
        if tr_a[i] < .07 or te_a[i] < .07:
            two_thirds = min(i, two_thirds)
            break
    tr_a_23 = tr_a[two_thirds:]
    te_a_23 = te_a[two_thirds:]
    x_axis_23 = x_axis[two_thirds:]

    #plt.title('Hidden Nodes = %s' % title)
    plt.subplot(212)
    plt.ylabel('0/1 Error (1-accuracy)')
    plt.xlabel('Half Epoch')
    plt.axis([two_thirds, epoch_count, 0, max(max(tr_a_23), max(te_a_23))])
    plt.plot(x_axis_23, tr_a_23, 'g^', x_axis_23, te_a_23, 'bs')
    plt.show()

if __name__ == '__main__':
    files = ['h10.txt', 'h50.txt', 'h100.txt', 'h500.txt', 'h1000.txt','rh500.txt']

    for filename in files:
        content = []
        with open(filename) as f:
            content = f.readlines()
            content = [x.strip() for x in content]

        tuples = [make_tuple(line) for line in content]

        title = filename.replace('.txt', '').replace('h','')
        plot_errors(tuples, title)
