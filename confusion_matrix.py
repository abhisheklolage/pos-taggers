#!/usr/bin/env python

import numpy as np

class ConfusionMatrix():
    def __init__(self):
        # labels to indexes
        self._labels = dict()
        # indexes to labels
        self._indexes = dict()

        self.tag_list = ['DET', 'NOUN', 'ADJ', 'VERB', 'ADP', '.', 'ADV', 'CONJ', 'PRT', 'PRON','NUM', 'X']
        for tidx in range(len(self.tag_list)):
            self._labels[self.tag_list[tidx]] = tidx
            self._indexes[tidx] = self.tag_list[tidx]
        # max index for labels
        self._midx = len(self.tag_list)
        # confusion matrix
        # plus 1 for sum
        self.cm = np.zeros(shape=(self._midx + 1, self._midx + 1), dtype=int)

    def clear(self):
        """
        Clears all counts in Confusion Matrix while
        leaving the existing labels in the class.
        """
        dim = self._midxi + 1
        self.cm = np.zeros((dim,dim),dtype=self._default_dtype)

    def reset(self):
        self.__init__()

    def add(self, grounds, preds):
        # add the gold and predicted labels
        # update the sizes of the CM and label,index lookups
        # then add value
        for ground,pred in zip(grounds, preds):
            # print (ground, pred)
            if len(ground) == len(pred):
                for glabel, plabel in zip(ground, pred):
                    g_idx = self._labels[glabel]
                    p_idx = self._labels[plabel]
                    # print (g_idx, p_idx)
                    # increasing the corresponding count
                    self.cm[g_idx][p_idx] += 1

        for row, col, idx in zip(self.cm, self.cm.T, range(len(self.cm))):
            self.cm[self._midx][idx] = col.sum()
            self.cm[idx][self._midx] = row.sum()
            idx += 1
        self.cm[self._midx][self._midx] = self.cm.sum()

    def pprint(self):
        """
        Pretty Print the Confusion Matrix with Labels
        """
        print(self.__str__())

    # default string output, makes pretty printed confusion matrix string
    def __str__(self):
        confusion_matrix_string = ""
        # max label length determines column spacing
        max_label_len = max(len(label) for label in self._labels)+ 5
        cmfmt = "{0:<"+str(max_label_len)+"}"
        # inital empty top-left column header
        confusion_matrix_string += cmfmt.format('')
        # print column header labels
        for i in range(0,len(self._indexes)):
            confusion_matrix_string += cmfmt.format(self._indexes[i])
        confusion_matrix_string += cmfmt.format('SUM')
        confusion_matrix_string += "\n"
        # print row labels and confusion matrix counts
        idx=0
        for i in range(self._midx + 1):
            for j in range(0,self._midx + 1):
                if j == 0:
                    if(idx == self._midx):
                        confusion_matrix_string += cmfmt.format('SUM')
                        confusion_matrix_string += cmfmt.format(self.cm[i,j])
                    else:
                        confusion_matrix_string += cmfmt.format(self._indexes[idx])
                        confusion_matrix_string += cmfmt.format(self.cm[i,j])
                    idx+=1
                else:
                    confusion_matrix_string += cmfmt.format(self.cm[i,j])
            confusion_matrix_string += "\n"
        return confusion_matrix_string
