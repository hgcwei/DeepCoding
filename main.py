from deepcoding import training_test_network
from common import utils
import time




def cross_validation(type):

    batch_sz = 500
    epoch_num = 3
    learning_rt = 1e-3

    tfrecords_ls = ['../data/'+ type + '/data_'+ type + '1.tfrecords', '../data/'+ type + '/data_'+ type + '2.tfrecords', '../data/'+ type + '/data_'+ type + '3.tfrecords']

    fprs = []
    tprs = []
    aucs = []
    time_elaspes = []

    # for i in range(1):
    for i in range(len(tfrecords_ls)):
        test_tfrecords_ls = [tfrecords_ls[i]]
        tfrecords_ls.pop(i)
        train_tfrecords_ls = tfrecords_ls
        time_elaspe, model_file, uuid_str = training_test_network.train_model(train_tfrecords_ls, test_tfrecords_ls, type, n_epoch=epoch_num,
                                                               learning_rate=learning_rt, batch_size= batch_sz)
        time.sleep(10)
        scores, labels = training_test_network.test_model(model_file, test_tfrecords_ls, batch_sz)
        print(scores)
        print(labels)
        fpr, tpr, auc = utils.eval_perf(labels, scores)
        utils.save_result('results/' + type + '/labels' + uuid_str + '.csv', labels)
        utils.save_result('results/' + type + '/scores' + uuid_str + '.csv', scores)
        utils.save_result('results/' + type + '/results' + uuid_str + '.csv', [tpr, 1 - fpr, auc, time_elaspe])


        # print(fpr,tpr,auc,time_elaspes)
        fprs.append(fpr)
        tprs.append(tprs)
        aucs.append(auc)
        time_elaspes.append(time_elaspe)

    return tprs, fprs, aucs, time_elaspes


# fpr_avg, tpr_avg, auc_avg, time_cost_avg = cross_validation('gm')
# fpr_avg, tpr_avg, auc_avg, time_cost_avg = cross_validation('th')
# fpr_avg, tpr_avg, auc_avg, time_cost_avg = cross_validation('tm')

type = 'tm'
# # tfrecords_ls = ['../data/' + type + '/data_' + type + '1.tfrecords',
# #                 '../data/' + type + '/data_' + type + '2.tfrecords',
# #                 '../data/' + type + '/data_' + type + '3.tfrecords',
# #                 '../data/' + type + '/data_' + type + '4.tfrecords',
# #                 '../data/' + type + '/data_' + type + '5.tfrecords']
model_file = 'model/tm/model_c58b46ddde524a98b1815fb1bbc01f3d.ckpt'
uuid_str = 'c58b46ddde524a98b1815fb1bbc01f3d'
scores, labels = training_test_network.test_model(model_file, ['test_tm2.tfrecords'], 500)
print(scores)
print(labels)
fpr, tpr, auc = utils.eval_perf(labels, scores)
utils.save_result('results/' + type + '/labels' + uuid_str + '.csv', labels)
utils.save_result('results/' + type + '/scores' + uuid_str + '.csv', scores)
utils.save_result('results/' + type + '/results' + uuid_str + '.csv', [tpr, 1 - fpr, auc])
# fpr, tpr, auc = train_test_model.test_model('model/th/model_68015182f83e483883b69401e393bf19.ckpt', [tfrecords_ls[0]], type,'68015182f83e483883b69401e393bf19')
#
#
#














