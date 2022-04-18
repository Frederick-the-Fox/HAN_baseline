import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from logreg import LogReg
import torch.nn as nn
import numpy as np
np.random.seed(0)
from sklearn.metrics import f1_score
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score
import pickle as pkl
import scipy.sparse as sp
import sys

def evaluate(embeds, ratio, idx_train, idx_val, idx_test, label, nb_classes, device, dataset, lr, wd
             , isTest=True):
    hid_units = embeds.shape[1]
    xent = nn.CrossEntropyLoss()

    train_embs = embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]

    train_lbls = torch.argmax(label[idx_train], dim=-1)
    val_lbls = torch.argmax(label[idx_val], dim=-1)
    test_lbls = torch.argmax(label[idx_test], dim=-1)
    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    auc_score_list = []

    for kkk in range(50):
        # print('range:{}'.format(kkk))
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)
        log.to(device)

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []

        logits_list = []
        for iter_ in range(200):
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

            # print(loss)

            # val
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            logits_list.append(logits)

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])
        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

        # auc
        best_logits = logits_list[max_iter]
        best_proba = softmax(best_logits, dim=1)
        auc_score_list.append(roc_auc_score(y_true=test_lbls.detach().cpu().numpy(),
                                            y_score=best_proba.detach().cpu().numpy(),
                                            multi_class='ovr'
                                            ))

    if isTest:
        print("\t[Classification] Macro-F1_mean: {:.4f} var: {:.4f}  Micro-F1_mean: {:.4f} var: {:.4f} auc {:.4f}"
              .format(np.mean(macro_f1s),
                      np.std(macro_f1s),
                      np.mean(micro_f1s),
                      np.std(micro_f1s),
                      np.mean(auc_score_list),
                      np.std(auc_score_list)
                      )
              )
    else:
        return np.mean(macro_f1s_val), np.mean(macro_f1s)

    f = open("result/result_"+dataset+str(ratio)+".txt", "a")
    f.write(str(np.mean(macro_f1s))+"\t"+str(np.mean(micro_f1s))+"\t"+str(np.mean(auc_score_list))+"\n")
    f.close()


if __name__ == '__main__':
    dataset = 'dblp'
    metapaths = 'APA,APCPA,APTPA'.split(",")#dblp:APA,APCPA,APTPA;freebase:MAM MDM MWM;imdb:MAM MDM MKM
    device = 'cpu'
    sc = 3.0
    data = pkl.load(open('/home/hangni/WangYC/HAN/data/mydata/'+dataset+'/'+dataset+'_new.pkl', "rb"))

    # print('train_idx:{}'.format(data['train_idx'].shape))

    label = data['label']
    idx_train = []
    idx_val = []
    idx_test = []
    ratio = [20, 40, 60]
    for i in range(len(ratio)):
        idx_train.append(data['train_idx_' + str(ratio[i])])
        idx_val.append(data['val_idx_' + str(ratio[i])])
        idx_test.append(data['test_idx_' + str(ratio[i])])


    # print('idx_train.shape:{}'.format(idx_train.shape))
    # print('idx_test.shape:{}'.format(idx_test.shape))
    # print('idx_val.shape:{}'.format(idx_val.shape))

    # labels = torch.FloatTensor(labels[np.newaxis])
    label = torch.FloatTensor(label)

    idx_train = [torch.LongTensor(i) for i in idx_train]
    idx_val = [torch.LongTensor(i) for i in idx_val]
    idx_test = [torch.LongTensor(i) for i in idx_test]

    # label = label.cuda()
    # idx_train = [i.cuda() for i in idx_train]
    # idx_val = [i.cuda() for i in idx_val]
    # idx_test = [i.cuda() for i in idx_test]


    embeds = np.load(dataset + sys.argv[1] + '.npy')
    print("embeds.shape:{}".format(embeds.shape))
    # result = validate(torch.tensor(embeds), 40, idx_train, idx_val, idx_test, label, 4, device, dataset, 0.01, 0)
    # print(result)
    # validate(embeds, 40, self.idx_eval_train, self.idx_eval_val, self.idx_eval_test, self.labels, nb_classes, self.args.device, self.args.dataset,
    #             self.args.eval_lr, self.args.eval_wd)

    for i in range(len(idx_train)):
        evaluate(torch.tensor(embeds), ratio[i], idx_train[i], idx_val[i], idx_test[i], label, 4, device, dataset,0.01, 0)#dblp:4