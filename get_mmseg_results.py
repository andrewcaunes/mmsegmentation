"""
Script made by Andrew Caunes.
Example use:
python script.py ...
"""
import os
import shutil
import argparse
import logging
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)
import numpy as np
import pandas as pd
from seg_3D_by_2D.classes_dicts import global_ext_classes_dict

def main(args):
    logging.info("args = %s", args)
    get_results(work_dirs_folder=args.work_dirs_folder,
                output_path=args.output_path,
                test_foldername=args.test_foldername)

def get_results(work_dirs_folder, 
                output_path, 
                test_foldername,
                classes_dict=None):
    """Given a work_dirs folder, search for a test_folder in each work dir and extract the results from the log file if found.
    The results are stored in a pandas datadrame (csv format) with the following columns:
    - work_dir
    - iter
    - mIoU
    - mAcc
    - aAcc
    - IoU_class_k
    for all classes from the classes dict"""
    if classes_dict is None:
        classes_dict = global_ext_classes_dict
    
    df = pd.DataFrame(columns=['work_dir', 'iter', 'mIoU', 'mAcc', 'aAcc'] + [f"IoU_{k}" for k in classes_dict.keys()])
    num_classes = len(classes_dict)

    work_dirs = os.listdir(work_dirs_folder)
    results = []
    for work_dir in work_dirs:
        test_folder = os.path.join(work_dirs_folder, work_dir, test_foldername)

        for root, dirs, files in os.walk(test_folder):
            if np.any([f.endswith('.json') for f in files]) and np.any([f.endswith('.log') for f in files]):
                # get iter
                root_dirs = root.split('/')
                assert np.any(['iter_' in d for d in root_dirs]), f"no root_dirs={root_dirs} contains iter_ "
                iter = [d for d in root_dirs if 'iter_' in d][0]
                iter = int(iter.split('_')[-1])
                logging.info(f"iter_ = {iter}")

                # check if work_dir and iter already in results
                if np.any([r['work_dir'] == work_dir and r['iter'] == iter for r in results]):
                    logging.info(f"work_dir = {work_dir} and iter = {iter} already in results")
                    continue

                # get log file
                log_file = [f for f in files if f.endswith('.log')][0]
                log_file_path = os.path.join(root, log_file)
                with open(log_file_path, 'r') as f:
                    lines = f.readlines()[-(num_classes+3):]

                # get IoU for each class
                metric_dict = {}
                for line in lines:
                    if not np.any([cls in line for cls in classes_dict.keys()]):
                        continue
                    line = [line.strip() for line in line.split('|') if line.strip() != '']
                    cls = line[0].replace(' ', '_')
                    metric_dict['IoU_'+cls] = float(line[1])
                    metric_dict['Acc_'+cls] = float(line[2])
                # get mIoU, mAcc, aAcc
                last_line= lines[-1].split(' ')
                assert 'aAcc:' in last_line and 'mIoU:' in last_line and 'mAcc:' in last_line, f"last_line={last_line}"
                for i in range(len(last_line)):
                    if last_line[i] == 'mIoU:':
                        metric_dict['mIoU'] = float(last_line[i+1])
                    elif last_line[i] == 'mAcc:':
                        metric_dict['mAcc'] = float(last_line[i+1])
                    elif last_line[i] == 'aAcc:':
                        metric_dict['aAcc'] = float(last_line[i+1])
                metric_dict['work_dir'] = work_dir
                metric_dict['iter'] = iter
                results.append(metric_dict)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    logging.info(f"Results saved in {output_path}")
    return df





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--work_dirs_folder', help='', default=None)
    parser.add_argument('--output_path', help='', default='results.csv')
    parser.add_argument('--test_foldername', help='', default='test_results')

    args = parser.parse_args()

    main(args)