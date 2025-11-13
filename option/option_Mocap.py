import os
import argparse
from pprint import pprint
from utils import other_utils as ou

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        """
        :return: option of project
        """

        "---basic option---"
        self.parser.add_argument('--dataset', type=str, default='Mocap', help='used dataset name')
        self.parser.add_argument('--ckpt', type=str, default='/home/s01040/ST-MoE/model/checkpoint', help='path of checkpoint')
        self.parser.add_argument('--tensorboard', type=str, default='./model/tensorboard/', help='path to save tensorboard log')
        self.parser.add_argument('--model', type=str, default='GCN_Mamba', help='model type used')
        self.parser.add_argument('--cudaid', type=int, default=0, help='cuda index used')
        self.parser.add_argument('--log_name', type=str, default=None, help='log name')

        "---hyperpara option---"
        self.parser.add_argument('--drop_out', type=float, default=0.1, help='drop out probability')
        self.parser.add_argument('--d_model', type=int, default=256, help='dimension of model GCN')
        self.parser.add_argument('--epoch', type=int, default=100)
        self.parser.add_argument('--batch_size', type=int, default=96)
        self.parser.add_argument('--test_batch', type=int, default=96)
        self.parser.add_argument('--lr_now', type=float, default=0.01)
        self.parser.add_argument('--lr_decay_rate', type=float, default=0.98)
        self.parser.add_argument('--max_norm', type=float, default=10000)
        self.parser.add_argument('--in_features', type=int, default=45, help='dim of input feature, n x j')
        self.parser.add_argument('--frame_in', type=int, default=50,
                                 help='input frame number used in dataloader')
        self.parser.add_argument('--frame_out', type=int, default=25,
                                 help='output frame number used in dataloader')
        self.parser.add_argument('--seq_len', type=int, default=75,
                                 help='frame number each sample')
        self.parser.add_argument('--num_experts', type=int, default=4,
                                 help='number of experts in Mamba')
        self.parser.add_argument('--top_k', type=int, default=1,
                                 help='topk of experts in Mamba')


        "---size option of module---"
        self.parser.add_argument('--num_stage', type=int, default=5, help='for GCN')
        self.parser.add_argument('--depth', type=int, default=3, help='the number of stmamba')
        self.parser.add_argument('--nlayer', type=int, default=1, help='the number of MoE_layers')


        "---execute option---"
        # self.parser.add_argument('--mode', type=str, default='train', help='mode of execute')
        self.parser.add_argument('--mode', type=str, default='train', help='mode of execute')
        self.parser.add_argument('--test_epoch', type=int, default=None,
                                 help='check the model with corresponding epoch')
        self.parser.add_argument('--save_results', type=bool, default=1,
                                 help='whether to save result')
        

        "---Parameter sensitivity experiment---"
        self.parser.add_argument('--w_sp', type=float, default=1, help='weight of loss of spatial prediction')
        self.parser.add_argument('--w_tp', type=float, default=1, help='weight of loss of temporal prediction')



    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self, makedir=True):
            self._initial()
            self.opt = self.parser.parse_args()

            if self.opt.model == 'STAGE_4':
                self.opt.d_model = 16
            
            if self.opt.log_name is None:
                if self.opt.dataset=='Mocap' or self.opt.dataset == 'Mix1' or self.opt.dataset == 'Mix2':
                    self.opt.log_name = 'exp_Mocap_{}_in{}_out{}_lr_{}_lrd_{}_bs_{}_ep_{}_gcn{}_mamba{}_loss_sp{}tp{}_dmodel{}_combine{}_nlayer{}_topk{}'.format(
                                                                        self.opt.model,
                                                                        self.opt.frame_in,
                                                                        self.opt.frame_out,
                                                                        self.opt.lr_now,
                                                                        self.opt.lr_decay_rate,
                                                                        self.opt.batch_size,
                                                                        self.opt.epoch,
                                                                        self.opt.num_stage,
                                                                        self.opt.depth,
                                                                        self.opt.w_sp,
                                                                        self.opt.w_tp,
                                                                        self.opt.d_model,
                                                                        self.opt.combine_mode,
                                                                        self.opt.nlayer,
                                                                        self.opt.top_k
                                                                        )
                else:
                    self.opt.log_name = 'exp_{}_{}_in{}_out{}_lr_{}_lrd_{}_bs_{}_ep_{}_gcn{}_mamba{}_loss_sp{}tp{}_dmodel{}_combine{}_nlayer{}_topk{}'.format(
                                                                        self.opt.dataset,
                                                                        self.opt.model,
                                                                        self.opt.frame_in,
                                                                        self.opt.frame_out,
                                                                        self.opt.lr_now,
                                                                        self.opt.lr_decay_rate,
                                                                        self.opt.batch_size,
                                                                        self.opt.epoch,
                                                                        self.opt.num_stage,
                                                                        self.opt.depth,
                                                                        self.opt.w_sp,
                                                                        self.opt.w_tp,
                                                                        self.opt.d_model,
                                                                        self.opt.combine_mode,
                                                                        self.opt.nlayer,
                                                                        self.opt.top_k
                                                                        )
            self.opt.exp = self.opt.log_name

            ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
            if makedir==True:
                if not os.path.isdir(ckpt):
                    os.makedirs(ckpt)
                    ou.save_options(self.opt, dataset="MO")
                self.opt.ckpt = ckpt
                ou.save_options(self.opt, dataset="MO")

            self._print()

            return self.opt