from __future__ import print_function, absolute_import
import time

from .utils.meters import AverageMeter


class RLIM_USL(object):
    def __init__(self, encoder, memory, alpha=0.999):
    # def __init__(self, *encoders, memories):
        super(RLIM_USL, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.alpha = alpha


    def train(self, epoch, data_loader1, num_cluster, optimizer, print_freq=10, train_iters=400):
        
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        
        losses2 = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs1 = data_loader1.next()
            
            data_time.update(time.time() - end)

            inputs, cams, indexes1 = self._parse_data(inputs1)
            
            bn_x = self._forward(inputs)
            
            loss2 = self.memory(bn_x, indexes1, cams, num_cluster, epoch)
            
            #loss = (loss1+loss2)
            loss = loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            losses.update(loss.item())
            #losses1.update(loss1.item())
            losses2.update(loss2.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ALL {:.3f} ({:.3f})\t'
                      #'Loss_view {:.3f} ({:.3f})\t'
                      'Loss_soft {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader1),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              #losses1.val, losses1.avg,
                              losses2.val, losses2.avg
                              ))
    

                
    def _parse_data(self, inputs):
        img, _, pids, proids, camid, indexes = inputs
        return img.cuda(), camid.cuda(), indexes.cuda()

    def _forward(self, inputs):
        
        bn_x = self.encoder(inputs)
        
        return bn_x