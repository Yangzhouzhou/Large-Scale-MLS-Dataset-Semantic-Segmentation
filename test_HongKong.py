# --*-- UTF-8- --*--
# Data  :  下午2:35
# Name  : test_HongKong.py
'''
 code for inference hongkong dataset using trained model
'''

import laspy
from helper_tool import ConfigHongKong
from HKSemNet import Network, compute_loss, IoUCalculator
from hongkong_test_dataset import HongKong, HongKongSampler
from utils.helper_ply import write_ply
import warnings
import numpy as np
import os, argparse
import time
import logging
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default='CHECKPOINT_PATH',
                    help='Model checkpoint path [default: None]')
parser.add_argument('--name', type=str, default='HongKong_inference', help='Name of the experiment')
parser.add_argument('--log_dir', default='test_output', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--gpu', type=int, default=0, help='which gpu do you want to use, -1 for cpu')
parser.add_argument('--format', default='.ply', help='format to save')
parser.add_argument('--config', type=str, default='HongKong', choices=['HongKong'], )
FLAGS = parser.parse_args()

# cereate class for inference
class Inference:
    def __init__(self):
        #################################################   log   #################################################
        self.log_dir = os.path.join(FLAGS.log_dir, FLAGS.name)
        if not os.path.exists(self.log_dir):
            os.makedirs(os.path.join(self.log_dir, 'val_preds'))  
        log_fname = os.path.join(self.log_dir, 'log_test_evaluate.txt')
        LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
        DATE_FORMAT = '%Y%m%d %H:%M:%S'
        logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT, datefmt=DATE_FORMAT, filename=log_fname)
        self.logger = logging.getLogger("Inference")
        #################################################   dataset   #################################################
        # dataset path
        self.path = 'DATASET_PATH'

        # get dataset config
        cfg = ConfigHongKong
        dataset = HongKong(path=self.path, mode='validation')
        test_dataset = HongKongSampler(dataset, 'validation')
        test_dataloader = DataLoader(test_dataset, batch_size=ConfigHongKong.val_batch_size, shuffle=True,
                                     collate_fn=test_dataset.collate_fn)
        self.cfg = cfg
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.test_dataloader = test_dataloader
        self.logger.info("======== HongKong Dataset Semantic Segmentation Prediction ===========")
        #################################################   model   #################################################
        # gpu setting
        if FLAGS.gpu >= 0:
            if torch.cuda.is_available():
                FLAGS.gpu = torch.device(f'cuda:{FLAGS.gpu:d}')
            else:
                warnings.warn('CUDA is not available on your machine. Running the algorithm on CPU.')
                FLAGS.gpu = torch.device('cpu')
        else:
            FLAGS.gpu = torch.device('cpu')
        self.device = FLAGS.gpu

        # network and optimizer
        self.net = Network(cfg)
        self.net.to(self.device)
        optimizer = optim.Adam(self.net.parameters(), lr=cfg.learning_rate)

        # load checkpoint
        checkpoint_path = FLAGS.checkpoint_path
        print(os.path.isfile(checkpoint_path))
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Model restored from %s" % checkpoint_path)
        else:
            raise ValueError('CheckPointPathError')
        #################################################   other   #################################################
        self.test_probs = self.init_probs()
        self.test_smooth = 0.95
        self.num_vote = 100
        self.is_save_per_prediction = True  # save per sample prediction


    def init_probs(self):
        probs = [np.zeros(shape=[l.shape[0], self.cfg.num_classes], dtype=np.float32)
                 for l in self.dataset.input_labels['validation']]
        return probs

    def test(self):
        self.logger.info("Start Testing")
        step_id = 0
        save_per_num = 20
        last_min = -0.5

        while last_min < self.num_vote:
            self.net.eval()     # set model to eval mode
            for batch_idx, batch_data in enumerate(self.test_dataloader):
                for key in batch_data:
                    if type(batch_data[key]) is list:
                        for i in range(len(batch_data[key])):
                            batch_data[key][i] = batch_data[key][i].to(self.device)
                    else:
                        batch_data[key] = batch_data[key].to(self.device)
                # Forward pass
                with torch.no_grad():
                    torch.cuda.synchronize()
                    end_points = self.net(batch_data)

                    if self.is_save_per_prediction:
                        if step_id % save_per_num == 0:
                            self.save_per_predicton(step_id, end_points)
                self.update_predict(end_points, batch_data)

                step_id += 1
            # new_min = 7.7
            new_min = np.min(self.test_dataset.min_possibility['validation'])
            self.logger.info('Test Epoch end. Min possibility = {:.1f}'.format(new_min))
            if last_min + 1 < new_min:
                # update last_min
                last_min += 1
                # get prediction result
                self.merge_and_store(format=FLAGS.format)

                return

            step_id = 0
            continue
        return

    def save_per_predicton(self, step_id, end_points):
        probs = end_points['logits'].transpose(1, 2).cpu().numpy()
        B = probs.shape[0]
        cloud_idx = end_points['cloud_inds'].cpu().numpy()

        for j in range(B):
            file_names = self.dataset.input_names['validation'][cloud_idx[j][0]]
            pred = np.argmax(probs[j], 1).astype(np.uint32)
            name = file_names + '_' + str(step_id) + '_' + str(j) + '.ply'
            pred_file = os.path.join(self.log_dir, 'sample_preds')
            if not os.path.exists(pred_file):
                os.makedirs(pred_file)
            xyz = end_points['features'][j, :, 0:3].cpu().numpy()
            write_ply(os.path.join(pred_file, name), [xyz, pred], ['x', 'y', 'z', 'pred',])

    def update_predict(self, end_points, batch_data):
        # store logits into list
        input_inds = end_points['input_inds'].cpu().numpy()
        cloud_idx = end_points['cloud_inds'].cpu().numpy()
        probs = end_points['logits'].transpose(1, 2).cpu().numpy()
        B = probs.shape[0]
        for j in range(B):
            prob = probs[j]
            inds = input_inds[j]
            c_i = cloud_idx[j][0]
            self.test_probs[c_i][inds] = self.test_probs[c_i][inds] * self.test_smooth + prob * (1 - self.test_smooth)

    def merge_and_store(self, format='.ply'):
        # initialize result directory
        root_dir = os.path.join(self.log_dir, 'predictions')
        show_dir = os.path.join(self.log_dir, 'show')
        os.makedirs(root_dir, exist_ok=True)
        os.makedirs(show_dir, exist_ok=True)

        # merge all predictions
        self.logger.info("****merge and store prediction")
        N = len(self.test_probs)
        for j in range(N):
            file_name = self.dataset.input_names['validation'][j]
            pred = np.argmax(self.test_probs[j], 1).astype(np.uint32)

            # save prediction
            pred_name = file_name + '_pred.txt'
            output_path = os.path.join(root_dir, pred_name)
            np.savetxt(output_path, pred, fmt='%d', delimiter='\t')
            print("\n **save testing result file: %s -> in %s" % (file_name, output_path))
            self.logger.info("save testing result file: %s -> in %s" % (file_name, output_path))

            xyz = self.dataset.input_trees['validation'][j].data
            xyz = np.array(xyz, dtype=np.float32)
            color = self.dataset.input_features['validation'][j][:, 0:3].astype(np.uint8)
            intensity = self.dataset.input_features['validation'][j][:, -1]
            intensity = intensity * 255

            # save sub point and pred
            if format == '.ply':
                pred_save_ply_file = os.path.join(show_dir, file_name + '_pred.ply')
                write_ply(pred_save_ply_file, [xyz, color, intensity, pred], ['x', 'y', 'z', 'red', 'green', 'blue',  'intensity','pred'])
                print("%s file pred done, saved in %s"%(file_name, pred_save_ply_file))
                self.logger.info("%s file pred done, saved in %s"%(file_name, pred_save_ply_file))

            elif format == '.las':
                pred_save_las_file = os.path.join(show_dir, file_name + '.laz')
                las_file = laspy.create(file_version="1.2", point_format=3)
                las_file.add_extra_dim(laspy.ExtraBytesParams(name="pred", type="uint8", description="Labels"))
                las_file.x, las_file.y, las_file.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
                las_file.red, las_file.green, las_file.blue = color[:, 0], color[:, 1], color[:, 2]
                las_file.intensity = intensity
                las_file.pred = pred
                las_file.write(pred_save_las_file)
                print("%s file pred done, saved in %s" % (file_name, pred_save_las_file))
                self.logger.info("%s file pred done, saved in %s" % (file_name, pred_save_las_file))

def main():
    tester = Inference()
    print("****** Start Testing *******")
    t0 = time.time()
    tester.test()
    t1 = time.time()
    d = t1 - t0
    print('Done. Time elapsed:', '{:.0f} s.'.format(d) if d < 60 else '{:.0f} min {:.0f} s.'.format(*divmod(d, 60)))



if __name__ == '__main__':
    main()







