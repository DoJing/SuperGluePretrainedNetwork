from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import cv2
import os
from models.matching import Matching
from models.superpoint import SuperPoint
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)

def save_mvg_points(pt,save_file):
    if os.path.exists(save_file):
        return
    with open(save_file, 'w') as file:
        for i in range(pt.shape[0]):
            st = str(pt[i][0])+" "+str(pt[i][1])+" "+'0'+" "+'0'+"\n"
            file.writelines(st)
        file.close()
def save_mvg_matches(matches,save_file):
    with open(save_file, 'w') as file:
        for key in matches.keys():
            file.writelines(key[0]+' '+key[1]+'\n')
            file.writelines(str(len(matches[key]))+'\n')
            for v in matches[key]:
                file.writelines(str(v[0])+' '+str(v[1])+'\n')
        file.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_pairs', type=str, default='pairs.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--input_dir', type=str, default='/home/dojing/fuwuqi/zhu/',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--output_dir', type=str, default='dump_match_pairs/',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')

    parser.add_argument(
        '--resize', type=int, nargs='+', default=[-1, -1],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=10000,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.001,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.05,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--viz', type=bool, default="true", help='Visualize the matches and dump the plots')

    opt = parser.parse_args()
    print(opt)



    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    with open(opt.input_pairs, 'r') as f:
        pairs = [l.split() for l in f.readlines()]



    # Load the SuperPoint and SuperGlue models.
    device = 'cuda'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)
    superpoint = SuperPoint(config.get('superpoint', {})).eval().to(device)
    # Create the output directories if they do not exist already.
    input_dir = Path(opt.input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))

    if opt.viz:
        print('Will write visualization images to',
              'directory \"{}\"'.format(output_dir))
    image_tensor = {}
    keypoints_tensor = {}
    matches_mvg = {}
    timer = AverageTimer(newline=True)
    for i, pair in enumerate(pairs):
        name0, name1, id0, id1 = pair[:4]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        viz_path = output_dir / '{}_{}_matches.png'.format(id0, id1)

        # Handle --cache logic.
        do_viz = opt.viz

        # If a rotation integer is provided (e.g. from EXIF data), use it:
        mvg_match_num = pair[-1]
        rot0, rot1 = 0, 0

        # Load the image pair.
        if id0 not in image_tensor.keys():
            _, inp0, _ = read_image(input_dir / name0, device, opt.resize, rot0, opt.resize_float)
            image_tensor[id0] = inp0
        if id1 not in image_tensor.keys():
            _, inp1, _ = read_image(input_dir / name1, device, opt.resize, rot1, opt.resize_float)
            image_tensor[id1] = inp1
        if image_tensor[id0] is None or image_tensor[id1] is None:
            print('Problem reading image pair: {} {}'.format(
                input_dir/name0, input_dir/name1))
            exit(1)
        timer.update('load_image')

        if id0 not in keypoints_tensor.keys():
            pred0 = superpoint({'image': image_tensor[id0]})
            keypoints_tensor[id0] = pred0
        if id1 not in keypoints_tensor.keys():
            pred1 = superpoint({'image': image_tensor[id1]})
            keypoints_tensor[id1] = pred1
        if keypoints_tensor[id0] is None or keypoints_tensor[id1] is None:
            print('Problem extract feature pair: {} {}'.format(
                input_dir/name0, input_dir/name1))
            exit(1)
        timer.update('detect key points')
        # Perform the matching.
        pred = matching({'image0': image_tensor[id0], 'image1': image_tensor[id1],'keypoints0':keypoints_tensor[id0],'keypoints1':keypoints_tensor[id1]})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        timer.update('matcher')
        save_mvg_points(kpts0, str(output_dir / name0)[:-4] + '.feat')
        save_mvg_points(kpts1, str(output_dir / name1)[:-4] + '.feat')

        index0 = np.where(matches > -1)[0]
        index1 = matches[index0]
        if len(index0) < min(len(kpts0),len(kpts1))*0.2:
            continue
        cur_match=[]
        for r, t in zip(index0, index1):
            cur_match.append((r, t))
        matches_mvg[(id0, id1)] = cur_match

        print("kpt num : ", len(kpts0), len(kpts1), len(index0), mvg_match_num)
        if do_viz:
            # Keep the matching keypoints.
            valid = matches > -1
            mkpts0 = kpts0[index0]
            mkpts1 = kpts1[index1]
            mconf = conf[valid]
            # Visualize the matches.
            color = cm.jet(mconf)
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0)),
            ]
            if rot0 != 0 or rot1 != 0:
                text.append('Rotation: {}:{}'.format(rot0, rot1))

            # Display extra parameter info.
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]

            image0 = cv2.imread(str(input_dir / name0), cv2.IMREAD_GRAYSCALE)
            image1 = cv2.imread(str(input_dir / name1), cv2.IMREAD_GRAYSCALE)
            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                text, viz_path, True,
                True, True, 'Matches', small_text)

            timer.update('viz_match')

        timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))

    save_mvg_matches(matches_mvg, str(output_dir/'matches.glue.txt'))


