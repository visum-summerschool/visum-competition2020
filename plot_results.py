import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import ast

FONTSIZE = 20
FONT = ImageFont.truetype('arial.ttf', FONTSIZE)
PRED_PATH = 'predictions.csv'
GT_PATH = '/home/master/dataset/test/labels.csv'
DATA_PATH = '/home/master/dataset/test/'
RESULTS_PATH = '/home/visum/results'

def load_img(seq_id, frame_id):
    path = os.path.join(DATA_PATH, 'seq' + seq_id, 'img' + frame_id + '.jpg')
    try:
        return Image.open(path).convert('RGB')
    except FileNotFoundError:
        print("%s does not exist...skiping frame" % path)
        return None

def get_score(df, seq_id, frame_id):
    return df[(df['seq'] == seq_id) & (df['frame'] == frame_id)]['score']

def load_bb(df, seq_id, frame_id):
    return df[(df['seq'] == seq_id) & (df['frame'] == frame_id)]['label']

def plot_frame(pred, gt, seq, frame, path):

    img = load_img(seq, frame)
    if(img is None):
        return

    draw = ImageDraw.Draw(img)

    pred_bb = load_bb(pred, seq, frame)
    gt_bb = load_bb(gt, seq, frame).iloc[0]
    scores = get_score(pred, seq, frame)

    # plot predicted bounding boxes (in blue)
    for bb, score in zip(pred_bb, scores):
        bb = ast.literal_eval(bb)
        xmin, ymin, xmax, ymax = bb[0], bb[1], bb[2], bb[3]
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=(0, 0, 255), width=3)
        draw.text((xmin, ymin - 20), str(np.round(score, 3)), fill=(0, 0, 255), font=FONT)
    
    # plot ground truth bounding boxes (in green)
    if(gt_bb):
        for xmin, ymin, xmax, ymax in gt_bb:
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=(0, 255, 0), width=3)

    # save image and create necessary paths
    img_path = os.path.join(path, 'seq' + seq)
    if(not os.path.exists(img_path)):
        os.makedirs(img_path)

    img_name = os.path.join(img_path, 'img' + frame + '.jpg') 
    img.save(img_name)

def plot_sequence(pred, gt, seq, path):

    frames = gt[gt['seq'] == seq]['frame']
    for frame in frames:
        plot_frame(pred, gt, seq, frame, path)


if __name__=='__main__':

    # Load predictions
    pred = pd.read_csv(PRED_PATH, delimiter=';', dtype={'seq': str, 'frame': str})

    # Load ground truth
    gt = pd.read_csv(GT_PATH, delimiter=';', dtype={'seq': str, 'frame': str})
    gt['label'] = gt['label'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else None)

    # plots one frame
    seq = '000'
    frame = '6'
    plot_frame(pred, gt, seq, frame, RESULTS_PATH)

    # plot all frames from all sequences
    seqs = gt['seq'].unique()

    # plots whole sequence
    for seq in seqs:
        print("processing seq %s" % seq)
        plot_sequence(pred, gt, seq, RESULTS_PATH)
    
