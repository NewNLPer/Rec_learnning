
import torch
import torch.nn as nn
import torch.nn.functional as F


class DIN():
    def __init__(self,candidate_em,history_list):
        """
        :param candadate_em: 候选物品em
        :param history_list: 历史交互物品em
        """
        super(DIN,self).__init__()
        self.candidata = candidate_em
        self.history_list = history_list

    def get_smi(self,em_1,em_2):
        similarity = F.cosine_similarity(em_1, em_2, dim=0)
        return similarity

    def forward(self):
        smi_list = []
        for item in self.history_list:
            smi_list.append(self.get_smi(self.candidata,item))
        return smi_list


candidata = torch.rand(10)
history_list = torch.rand(5,10)


din = DIN(candidata,history_list)
res = din.forward()
print(res)