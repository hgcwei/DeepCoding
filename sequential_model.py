import numpy as np

def obtain_c2_feature_for_a_list_of_sequences(seqs):
  c2_features = []
  for seq in seqs:
      this_kmer_feature = obtain_c2_feature_for_one_sequence(seq)
      c2_features.append(this_kmer_feature)
  return np.hstack((c2_features))

def obtain_c2_feature_for_one_sequence(seq):
  data = np.zeros((len(seq),2),dtype=np.uint8)
  for i in range(len(seq)):
        if seq[i] == 'A':
            data[i] = [0,0]
        if seq[i] == 'C':
            data[i] = [1,1]
        if seq[i] == 'G':
            data[i] = [1,0]
        if seq[i] == 'T':
            data[i]= [0,1]
  return data
