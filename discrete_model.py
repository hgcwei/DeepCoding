import numpy as np
class kmer_featurization:

  def __init__(self, l, k):
    """
    seqs: a list of DNA sequences
    k: the "k" in k-mer
    """
    self.k = k
    self.l = l
    self.letters = ['A', 'C', 'G', 'T']
    # self.letters = ['A', 'T', 'C', 'G']
    self.multiplyBy = 4 ** np.arange(k-1, -1, -1) # the multiplying number for each digit position in the k-number system
    self.n = 4**k # number of possible k-mers

  def obtain_kmer_feature_for_a_list_of_sequences(self, seqs, write_number_of_occurrences=False):
    """
    Given a list of m DNA sequences, return a 2-d array with shape (m, 4**k) for the 1-hot representation of the kmer features.

    Args:
      write_number_of_occurrences:
        a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.
    """
    kmer_features = []
    for seq in seqs:
      this_kmer_feature = self.obtain_kmer_feature_for_one_sequence(seq.upper(), write_number_of_occurrences=write_number_of_occurrences)
      kmer_features.append(this_kmer_feature)

    kmer_features = np.array(kmer_features,dtype=np.uint8)

    return kmer_features

  def obtain_kmer_feature_for_one_sequence(self, seq, write_number_of_occurrences=False):
    """
    Given a DNA sequence, return the 1-hot representation of its kmer feature.

    Args:
      seq:
        a string, a DNA sequence
      write_number_of_occurrences:
        a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.
    """
    number_of_kmers = len(seq) - self.k + 1

    kmer_feature = np.zeros(self.n,dtype=np.uint8)

    for i in range(number_of_kmers):
      this_kmer = seq[i:(i+self.k)]
      this_numbering = self.kmer_numbering_for_one_kmer(this_kmer)
      kmer_feature[this_numbering] += 1

    if not write_number_of_occurrences:
      kmer_feature = kmer_feature / number_of_kmers

    return kmer_feature

  def obtain_frame_sensitive_kmer_feature_for_a_list_of_sequences(self, seqs, write_number_of_occurrences=False):
    """
    Given a list of m DNA sequences, return a 2-d array with shape (m, 4**k) for the 1-hot representation of the kmer features.

    Args:
      write_number_of_occurrences:
        a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.
    """
    kmer_features = []
    for seq in seqs:
      this_kmer_feature = self.obtain_frame_sensitive_kmer_feature_for_one_sequence(seq.upper(), write_number_of_occurrences=write_number_of_occurrences)
      kmer_features.append(this_kmer_feature)

    kmer_features = np.array(kmer_features,dtype=np.uint8)

    return kmer_features

  def obtain_frame_sensitive_kmer_feature_for_one_sequence(self, seq, write_number_of_occurrences=False):
    """
    Given a DNA sequence, return the 1-hot representation of its kmer feature.

    Args:
      seq:
        a string, a DNA sequence
      write_number_of_occurrences:
        a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.
    """
    number_of_kmers = len(seq) - self.k + 1

    kmer_feature = np.zeros(self.n,dtype=np.uint8)

    for i in range(0,number_of_kmers,3):
      this_kmer = seq[i:(i+self.k)]
      this_numbering = self.kmer_numbering_for_one_kmer(this_kmer)
      kmer_feature[this_numbering] += 1

    if not write_number_of_occurrences:
      kmer_feature = kmer_feature / number_of_kmers

    return kmer_feature

  def kmer_numbering_for_one_kmer(self, kmer):
    """
    Given a k-mer, return its numbering (the 0-based position in 1-hot representation)
    """
    digits = []
    for letter in kmer:
      digits.append(self.letters.index(letter))

    digits = np.array(digits)

    numbering = (digits * self.multiplyBy).sum()

    return numbering

  def kmer_numbering_for_one_gapped_kmer(self,lseq):
    x= 0
    pos = np.zeros(10,np.uint16)
    for i in range(3):
        for j in range(i+1,4):
            for k in range(j+1,5):
              substr = lseq[i]+lseq[j]+lseq[k]
              y = self.kmer_numbering_for_one_kmer(substr)
              pos[x] = y*10 + x
              x = x + 1
    return pos

  def obtain_frame_sensitive_gapped_kmer_feature_for_a_list_of_sequences(self, seqs, write_number_of_occurrences=False):
    """
    Given a list of m DNA sequences, return a 2-d array with shape (m, 4**k) for the 1-hot representation of the kmer features.

    Args:
      write_number_of_occurrences:
        a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.
    """
    kmer_features = []
    for seq in seqs:
      this_kmer_feature = self.obtain_frame_sensitive_gapped_kmer_feature_for_one_sequence(seq.upper(), write_number_of_occurrences=write_number_of_occurrences)
      kmer_features.append(this_kmer_feature)

    kmer_features = np.array(kmer_features,dtype=np.uint8)

    return kmer_features

  def obtain_frame_sensitive_gapped_kmer_feature_for_one_sequence(self, seq, write_number_of_occurrences=False):
    number_of_kmers = len(seq) - self.k

    kmer_feature = np.zeros(640, dtype=np.uint8)

    for i in range(0, number_of_kmers, 3):
      this_kmer = seq[i:(i + self.l)]
      this_numbering = self.kmer_numbering_for_one_gapped_kmer(this_kmer)
      kmer_feature[this_numbering] += 1

    if not write_number_of_occurrences:
      kmer_feature = kmer_feature / number_of_kmers

    return kmer_feature

#
# obj = kmer_featurization(5,3)
# # print(obj.kmer_numbering_for_one_kmer('ATA'))
#
# # print(obj.kmer_numbering_for_one_gapped_kmer('ATATA'))
# seq = 'ATATAA'
# print(seq[0:5])
# print(obj.obtain_frame_sensitive_kmer_feature_for_one_sequence('ATATAA',write_number_of_occurrences=True))
# x = obj.obtain_frame_sensitive_gapped_kmer_feature_for_a_list_of_sequences(['ATATAA','ATAAAA','TCAAAA'],write_number_of_occurrences=True)
#
# print(x.shape)
#
#
#
