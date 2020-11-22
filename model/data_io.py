__author__ = 'Administrator'
import numpy as np
import os
import glob
from sklearn.metrics import auc,roc_curve

def get_pos_neg_inds(lab,ratio):
    ninds = np.where(lab<1)
    pinds = np.where(lab>0)
    num = int(np.min([len(ninds[0]),len(pinds[0])])*ratio)+1
    rn = np.random.permutation(len(ninds[0]))
    rp = np.random.permutation(len(pinds[0]))
    return np.hstack((ninds[0][rn[0:num]],pinds[0][rp[0:num]]))

def get_close_inds(ind,l,flag):
    if flag == 1:
        return ind
    if flag == 2:
        inds = [ind -42, ind, ind+42]
    elif flag == 0:
        # inds = [ind-168,ind-126,ind-84,ind-42,ind,ind+42,ind+84,ind+126,ind+168]
        inds = [ind-180,ind-135,ind-90,ind-45,ind,ind+45,ind+90,ind+135,ind+180]
    elif flag == 3:
        inds = [ind-168,ind-147,ind-126,ind-105,ind-84,ind-63,ind-42,ind-21,ind,ind+21,ind+42,ind+63,ind+84,ind+105,ind+126,ind+147,ind+168]
    for i in range(len(inds)):
        if inds[i] <0:
           inds[i]=0
        elif inds[i]> l-1:
           inds[i]=l-1
    return inds

# def gen_one_x(inds,data):
#     c = data[:,0:2]
#     l = len(c)
#     vss = []
#     for i in range(len(inds)):
#         idxs = get_close_inds(inds[i],l)
#         vs = np.hstack((c[idxs,:]))
#         vss.append(vs)
#     return np.asarray(vss, np.float)

def gen_one_x(inds,data,flag):
    l = len(inds)
    # one_x = np.zeros((l,18))
    c = data[:,0:-1]
    one_x = []
    for i in range(l):
        idxs = get_close_inds(inds[i],len(c),flag)
        # one_x[i,:] = np.hstack((c[idxs,:]))
        one_x.append(np.hstack((c[idxs,:])))
    return np.matrix(one_x)

def gen_one_data(data,ratio,flag):
    lab = data[:,-1]
    inds = get_pos_neg_inds(lab,ratio)
    one_y = lab[inds]
    one_x = gen_one_x(inds,data,flag)
    return one_x, one_y

def gen_data(csvs,ratio,flag):
    x = []
    y = []
    for i in range(len(csvs)):
        data = csvs[i]
        one_x, one_y = gen_one_data(data,ratio,flag)
        x.append(one_x)
        y.append(one_y)
    return np.concatenate(x,axis=0), np.concatenate(y,axis=0)

def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

def minibatches2(inputs=None, batch_size=None, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt]


def read_all_csv(path):
    csvs = []
    files = os.listdir(path)  # 采用listdir来读取所有文件
    files.sort(key=lambda x:int(x[:-4]))
    for file_ in files:
        if not os.path.isdir(path + file_):
            csv = read_one_csv(path+file_)
            csvs.append(csv)
    return csvs

def read_one_csv(file):
    csv = np.loadtxt(file,delimiter=',')
    return csv

def gen_tr_te_data_from_csvs(path,ratio,flag):
    x = []
    y = []
    err = []
    files = os.listdir(path)  # 采用listdir来读取所有文件
    files.sort(key=lambda x:int(x[:-4]))
    for file_ in files:
        if not os.path.isdir(path + file_):
            # csv = np.loadtxt(path+file_, delimiter=',')
            try:
                csv = read_one_csv(path + file_)
                one_x, one_y = gen_one_data(csv,ratio,flag)
                x.append(one_x)
                y.append(one_y)
            except:
                err.append(file_)

    return np.concatenate(x,axis=0), np.concatenate(y,axis=0),err

def read_two_csvs(f1,f2):
    csv1 = read_one_csv(f1)
    csv2 = read_one_csv(f2)
    return np.hstack((csv1[:,0:-1],csv2))

def gen_tr_te_data_from_two_csvs(path1,path2,file_ls,ratio,flag):
    x = []
    y = []
    err = []
    for i in range(len(file_ls)):
        f1 = path1 + str(file_ls[i]) + '.csv'
        f2 = path2 + str(file_ls[i]) + '.csv'
        if not os.path.isdir(f1) and not os.path.isdir(f2):
            try:
                csv = read_two_csvs(f1,f2)
                one_x,one_y = gen_one_data(csv,ratio,flag)
                x.append(one_x)
                y.append(one_y)
            except:
                err.append(i)
    return np.concatenate(x,axis=0), np.concatenate(y,axis=0),err

def read_csv(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    csvs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.csv'):
            # print('reading the images:%s' % (im))
            # img = mpimg.imread(im)
            mat = np.loadtxt(open(im, "rb"), delimiter=",", skiprows=0)
            # img = transform.resize(img, (w, h))
            csvs.append(mat)
            labels.append(idx)
    return np.asarray(csvs, np.float32), np.asarray(labels, np.int32)

def save_csv(fn,v):
    np.savetxt(fn, v, delimiter=",")

def gen_samples_for_orth(data):
    ds = data.shape
    samples = np.zeros((ds[0],91,4,1))
    data_ext = np.vstack((np.zeros((45,4)),data,np.zeros((45,4))))
    for i in range(ds[0]):
        samples[i,:,:,:] = np.reshape(data_ext[i:i+91,:],(91,4,1))
    return samples

def eval_perf(label,scores):
    fpr, tpr, thresholds = roc_curve(label, scores)
    for i in range(len(thresholds)):
        if thresholds[i] < 0.5:
            return fpr[i],tpr[i],auc(fpr,tpr)






