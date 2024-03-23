import os
import pickle
import torch
from torch.utils.data import DataLoader
import loader
from model import MultiModelModel
from utils import seed_worker, seed_everything, train, evaluate
if __name__ == '__main__':
    num_workers = 8
    #bert模型
    encoder_t = 'bert-base-uncased'
    #Resnet152模型加载
    encoder_v = 'resnet152'
    #twitter推文数据库
    dataset = 'twitter2015'
    #学习速率
    lr = 1e-5
    #学习迭代次数
    num_epochs = 1
    #Adam优化器
    optim = 'Adam'
    #32 Batch Size
    bs = 16
    #seed为0
    seed_everything(0)
    generator = torch.Generator()
    generator.manual_seed(0)

    if num_workers > 0:
        torch.multiprocessing.set_sharing_strategy('file_system')
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    #加载数据
    ner_corpus = loader.load_ner_corpus(f'resources/datasets/{dataset}', load_image=(encoder_v != ''))
    ner_train_loader = DataLoader(ner_corpus.train, batch_size=bs, collate_fn=list, num_workers=num_workers,
                                  shuffle=True, worker_init_fn=seed_worker, generator=generator)
    ner_dev_loader = DataLoader(ner_corpus.dev, batch_size=bs, collate_fn=list, num_workers=num_workers)
    ner_test_loader = DataLoader(ner_corpus.test, batch_size=bs, collate_fn=list, num_workers=num_workers)
    #新建模型
    model = MultiModelModel.from_pretrained(0, encoder_t, encoder_v) #CUDA编号,Transformer Encoder, Vision Encoder
    #初始化参数
    params = [
        {'params': model.encoder_t.parameters(), 'lr': lr},
        {'params': model.head.parameters(), 'lr': lr * 100},
        {'params': model.gcn_layer.parameters(), 'lr': lr*100},
        {'params': model.encoder_v.parameters(), 'lr': lr},
        {'params': model.proj.parameters(), 'lr': lr * 100},
        {'params': model.rnn.parameters(), 'lr': lr * 100},
        {'params': model.crf.parameters(), 'lr': lr * 100},
        {'params': model.aux_head.parameters(), 'lr': lr * 100}
    ]

    optimizer = getattr(torch.optim, optim)(params)

    dev_f1s, test_f1s = [], []
    ner_losses, itr_losses = [], []
    best_dev_f1, best_test_report = 0, None
    #训练
    for epoch in range(1, num_epochs + 1):

        ner_loss = train(ner_train_loader, model, optimizer, task='ner')
        ner_losses.append(ner_loss)
        print(f'#{epoch}迭代实体识别loss: {ner_loss:.2f}')

        dev_f1, dev_report = evaluate(model, ner_dev_loader)
        dev_f1s.append(dev_f1)
        test_f1, test_report = evaluate(model, ner_test_loader)
        test_f1s.append(test_f1)
        print(f'f1分数-dev数据库: {dev_f1:.4f}, f1分数-test数据库: {test_f1:.4f}')
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_report = test_report
    print()
    #输出
    print(best_test_report)

    #保存训练模型
    file_name = f'trained/{encoder_t}-BiLSTM-{encoder_v}.pkl'
    pickle.dump(model, open(file_name, 'wb'))
