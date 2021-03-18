import tensorflow as tf
from discrete_model import kmer_featurization
import sequential_model

def count_tfrecord_number(tf_records_ls):
    c = 0
    for fn in tf_records_ls:
        for record in tf.python_io.tf_record_iterator(fn):
            c += 1
    return c

def parse_tfrecord(filename_ls):
    filename_queue = tf.train.string_input_producer(filename_ls, shuffle=True)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
          'c2_': tf.FixedLenFeature([], tf.string),
          'gkm_': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64)
                    }
            )
    return tf.reshape(tf.decode_raw(features['c2_'],tf.uint8),[90,6,1]),tf.reshape(tf.decode_raw(features['gkm_'],tf.uint8),[640,3,1]), features['label']

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def encode_one_sample(line,ws):
     l = len(line)
     seq_list = []
     for i in range(int(l/ws)):
         seq_list.append(line[i*ws:i*ws+ws])
     obj = kmer_featurization(5,3)
     kmer_features = obj.obtain_frame_sensitive_gapped_kmer_feature_for_a_list_of_sequences(seq_list, write_number_of_occurrences=True)
     c2_features = sequential_model.obtain_c2_feature_for_a_list_of_sequences(seq_list)
     return c2_features,kmer_features.T,int(line[l-1])

def samples2tfRecord(filename,recordname,ws):
    f = open(filename)
    writer = tf.python_io.TFRecordWriter(recordname)
    for line in f.readlines():
        line = line.strip('\n')
        c2,gkm,label = encode_one_sample(line,ws)
        c2_ = c2.tostring()
        gkm_ = gkm.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
                'c2_':_bytes_feature(c2_),
                'gkm_': _bytes_feature(gkm_),
                'label': _int64_feature(label)
                }))
        writer.write(example.SerializeToString())
    writer.close()
    return filename

# samples2tfRecord('D:/matlab_projs/code_opt/code_opt2/data_tm1.txt','data_tm1.tfrecords',90)
# samples2tfRecord('D:/matlab_projs/code_opt/code_opt2/data_tm2.txt','data_tm2.tfrecords',90)
# samples2tfRecord('D:/matlab_projs/code_opt/code_opt2/data_tm3.txt','data_tm3.tfrecords',90)

# samples2tfRecord('D:/matlab_projs/code_opt/code_opt2/test_tm2.txt','test_tm2.tfrecords',90)
# samples2tfRecord('D:/matlab_projs/code_opt/code_opt2/data_th2.txt','data_th2.tfrecords',90)
# samples2tfRecord('D:/matlab_projs/code_opt/code_opt2/data_th3.txt','data_th3.tfrecords',90)

# samples2tfRecord('D:/matlab_projs/code_opt/code_opt2/data_gh1.txt','data_gh1.tfrecords',90)
# samples2tfRecord('D:/matlab_projs/code_opt/code_opt2/data_gh2.txt','data_gh2.tfrecords',90)
# samples2tfRecord('D:/matlab_projs/code_opt/code_opt2/data_gh3.txt','data_gh3.tfrecords',90)

# samples2tfRecord('D:/matlab_projs/code_opt/code_opt2/data_gm1.txt','data_gm1.tfrecords',90)
# samples2tfRecord('D:/matlab_projs/code_opt/code_opt2/data_gm2.txt','data_gm2.tfrecords',90)
# samples2tfRecord('D:/matlab_projs/code_opt/code_opt2/data_gm3.txt','data_gm3.tfrecords',90)
