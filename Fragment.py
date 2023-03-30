import mne
from mne.time_frequency import psd_welch
from mne.viz import plot_topomap
from mne.io import read_raw_edf

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import pandas as pd

import os
import json
from PIL import Image

import UIPanels


class EEGRecordsSet:
    eeg_classes = {'FoDo': 0, 'FzDo': 1}

    records = []
    participant_name = ''

    def __init__(self, input_folder):
        with open(os.path.join(input_folder, 'Classes.json')) as json_file:
            self.eeg_classes = json.load(json_file)
        self.records = [None for i in range(len(self.eeg_classes))]

        self.participant_name = input_folder.split('\\')[-1]

        filenames = os.listdir(input_folder)
        for filename in filenames:
            if filename.endswith('.EDF'):
                saved_data_dir = os.path.join('saved_files/fragments_data', self.participant_name)
                record = EEGRecord(os.path.join(input_folder, filename), self.eeg_classes,
                                   saved_data_dir if os.path.exists(saved_data_dir) else None)
                self.records[self.eeg_classes[record.get_record_name()]] = record

        self.save_all()

    def get_records_number(self):
        return len(self.records)

    def get_records_names(self):
        return list(self.eeg_classes.keys())

    def get_record(self, idx):
        return self.records[idx]

    def get_record_by_name(self, name):
        return self.records[self.get_records_names().index(name)]

    def get_max_fragments_number(self):
        return max([record.get_fragments_number() for record in self.records])

    def get_chs(self):
        return self.records[0].get_chs()

    def get_tg_imgs(self, band):
        imgs = []
        labels = []
        for record in self.records:
            imgs.append([])
            labels.append([])

            for idx in range(record.get_fragments_number()):
                if band != -1:
                    imgs[-1].append(record.get_fragment_tg_img(idx, band))
                else:
                    imgs[-1].append(record.get_fragment_tg_img_all_bands(idx))
                labels[-1].append(record.get_fragment(0, idx).get_record_name() + str(idx))

        return imgs, labels

    def get_ml_tg_imgs(self):
        print(self.participant_name)

        imgs, labels, classes = [], [], []

        for record in self.records:
            for idx in range(record.get_fragments_number()):
                imgs.append(record.get_fragment_tg_img_all_bands(idx))
                labels.append(self.participant_name + '_' + record.get_record_name() + '_' + str(idx) + '.jpg')
                classes.append(record.get_record_class())

        df_data = [[labels[i], classes[i]] for i in range(len(labels))]
        df = pd.DataFrame(df_data, columns=['file_path', 'label_group'])

        return imgs, df

    def get_ml_data(self):
        ml_data = []
        for record in self.records:
            for fragment in record.get_all_fragments():
                if fragment.get_status():
                    ml_data.append(fragment.get_ml_vec())

        return ml_data

    def save_all(self):
        dir = os.path.join('saved_files/fragments_data', self.participant_name)
        if not os.path.exists(dir):
            os.mkdir(dir)
            for record in self.records:
                record.save_record(dir)


class EEGRecord:
    chs_number = 31
    offset = 10
    fragment_length = 32

    filename = None
    raw, rec_class, rec_name = None, None, None
    fragments = [[]]

    vis_raw = None

    def __init__(self, path, eeg_classes, saved_data_dir=None):
        self.raw = self.read_eeg(path)
        self.vis_raw = self.raw.copy()

        self.filename = os.path.basename(path)
        postfix = self.filename.split('.')[0].split('_')[1]
        self.rec_name = postfix
        self.rec_class = eeg_classes[postfix]

        interval_length = self.fragment_length
        tmin = 0
        duration = self.raw.times[-1]
        intervals_number = int((duration - tmin) // interval_length)
        t_intervals = [(tmin + i * interval_length,
                        tmin + (i + 1) * interval_length)
                       for i in range(intervals_number)]

        def get_other_chs(idx):
            return lambda: self.get_fragments_by_idx(idx)

        data = None
        if saved_data_dir is not None:
            path = os.path.join(saved_data_dir, self.rec_name)
            with open(path) as json_file:
                data = json.load(json_file)

        self.fragments = [
            [Fragment(self.raw, ch, idx, self.rec_class, self.rec_name, t_intervals[idx][0], t_intervals[idx][1],
                      get_other_chs(idx), data)
             for idx in range(len(t_intervals))]
            for ch in range(self.chs_number)]

    def read_eeg(self, path):
        raw = read_raw_edf(path, eog=(), preload=True)
        raw.describe()

        raw.drop_channels(raw.ch_names[31:])
        raw.crop(tmin=self.offset)

        montage_design = ["Fp1", "Fpz", "Fp2", "F7", "F3", "Fz",
                          "F4", "F8", "FT7", "FC3", "FCz", "FC4", "FT8",
                          "T3", "C3", "Cz", "C4", "T4", "TP7",
                          "CP3", "CPz", "CP4", "TP8", "T5",
                          "P3", "Pz", "P4", "T6", "O1", "Oz", "O2"]
        new_ch_names = {raw.ch_names[i]: montage_design[i] for i in range(len(raw.ch_names))}
        raw.rename_channels(new_ch_names)
        raw.describe()

        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)

        return raw

    def get_fragments_number(self):
        return len(self.fragments[0])

    def get_fragment(self, ch, idx):
        return self.fragments[ch][idx]

    def get_fragments_by_idx(self, idx):
        return np.array(self.fragments)[:, idx].tolist()

    def get_fragments_by_ch(self, ch):
        return self.fragments[ch]

    def get_all_fragments(self):
        return np.array(self.fragments).flatten().tolist()

    def get_chs(self):
        return self.raw.ch_names

    def get_record_name(self):
        return self.rec_name

    def get_record_class(self):
        return self.rec_class

    def get_fragment_tg(self, idx, band):
        band_power = [self.get_fragment(i, idx).get_parts_avg_adjusted()[band] for i in range(self.chs_number)]
        return band_power

    def get_fragment_tg_img(self, idx, band):
        fig, ax = plt.subplots(figsize=(3, 3))
        band_power = self.get_fragment_tg(idx, band)
        plot_topomap(band_power, self.raw.info, axes=ax, cmap=cm.viridis, show=False)
        img = UIPanels.PanelImg(fig).draw()
        return img

    def get_fragment_tg_img_all_bands(self, idx):
        de = np.array(self.get_fragment_tg(idx, 2))
        th = np.array(self.get_fragment_tg(idx, 3))
        deth = de + th
        a1 = np.array(self.get_fragment_tg(idx, 4))
        a2 = np.array(self.get_fragment_tg(idx, 5))
        a1a2 = a1 + a2
        b1 = np.array(self.get_fragment_tg(idx, 6))
        b2 = np.array(self.get_fragment_tg(idx, 7))
        b1b2 = b1 + b2

        fig, ax = plt.subplots(num=0, figsize=(5.12, 5.12))
        plot_topomap(deth, self.raw.info, axes=ax, cmap=cm.gray, show=False)
        deth_img = UIPanels.PanelImg(fig).draw()
        fig, ax = plt.subplots(num=0, figsize=(5.12, 5.12))
        plot_topomap(a1a2, self.raw.info, axes=ax, cmap=cm.gray, show=False)
        a1a2_img = UIPanels.PanelImg(fig).draw()
        fig, ax = plt.subplots(num=0, figsize=(5.12, 5.12))
        plot_topomap(b1b2, self.raw.info, axes=ax, cmap=cm.gray, show=False)
        b1b2_img = UIPanels.PanelImg(fig).draw()

        r = np.asarray(deth_img.convert('L'))
        g = np.asarray(a1a2_img.convert('L'))
        b = np.asarray(b1b2_img.convert('L'))

        rgb_img_data = np.stack((r, g, b))
        rgb_img_data = np.moveaxis(rgb_img_data, 0, -1)

        rgb_img = Image.fromarray(rgb_img_data)

        plt.pause(0.001)

        return rgb_img

    def draw_psd(self, ax):
        mne.viz.plot_raw_psd(self.vis_raw, fmin=0.25, fmax=40, tmin=self.offset, n_fft=8 * 256, ax=ax)

    def hide_ch(self, ch):
        self.vis_raw.drop_channels(self.get_chs()[ch])

    def save_record(self, dir):
        data = {fragment.get_code(): fragment.get_parts_data()
                for fragment in np.array(self.fragments).flatten()}

        path = os.path.join(dir, self.rec_name)
        with open(path, 'w') as fp:
            json.dump(data, fp)


class Fragment:
    raw, ch, idx, rec_class, rec_name = None, None, None, None, None
    tmin, tmax = None, None
    parts_number = 4

    parts = []
    status = True

    get_other_chs = None

    def __init__(self, raw, ch, idx, rec_class, rec_name, tmin, tmax, get_other_chs, data=None):
        self.raw, self.ch, self.idx, self.rec_class, self.rec_name = raw, ch, idx, rec_class, rec_name
        self.tmin, self.tmax = tmin, tmax
        self.get_other_chs = get_other_chs

        interval_length = (self.tmax - self.tmin) / self.parts_number
        t_intervals = [(self.tmin + i * interval_length,
                        self.tmin + (i + 1) * interval_length)
                       for i in range(self.parts_number)]

        parts_data = data[self.get_code()] if data is not None else None
        self.parts = [Part(self.raw, self.ch, t_intervals[i][0], t_intervals[i][1],
                           parts_data[i] if parts_data is not None else None)
                      for i in range(len(t_intervals))]

    def set_status(self, status):
        self.status = status

    def get_status(self):
        return self.status

    def get_norm_parts_data(self):
        return [part.get_norm_params() for part in self.parts]

    def get_norm_parts_avg(self):
        return np.mean(self.get_norm_parts_data(), 0)

    def get_norm_parts_rng(self):
        return np.ptp(self.get_norm_parts_data(), 0)

    def get_parts_data(self):
        return [part.get_params() for part in self.parts]

    def get_parts_avg(self):
        return np.mean(self.get_parts_data(), 0)

    def get_parts_avg_adjusted(self):
        if self.status:
            return self.get_parts_avg()
        else:
            data = []
            valid_chs_pos = []
            chs = self.get_other_chs()
            for ch in chs:
                if ch.get_status():
                    data.append(ch.get_parts_avg())
                    valid_chs_pos.append(ch.get_ch_pos())

            pos = self.get_ch_pos()
            dist = [np.linalg.norm(ch_pos - pos) for ch_pos in valid_chs_pos]
            data = [x for _, x in sorted(zip(dist, data))]
            data = data[:3]

            avg = np.mean(data, 0)
            return avg

    def get_parts_rng(self):
        return np.ptp(self.get_parts_data(), 0)

    def get_ch_pos(self):
        return list(self.raw.get_montage().get_positions()['ch_pos'].values())[self.ch]

    def get_ml_vec(self):
        return list(self.get_ch_pos()) + list(np.reshape(self.get_norm_parts_data(), -1)) + [self.rec_class]

    def get_raw(self):
        return self.raw

    def get_record_name(self):
        return self.rec_name

    def get_code(self):
        return str(self.ch) + '_' + str(self.idx)

    def get_tmin(self):
        return self.tmin

    def get_tmax(self):
        return self.tmax

    def get_chs(self):
        return self.raw.ch_names

    def get_ch_name(self):
        return self.raw.ch_names[self.ch]

    def draw_psd(self, ax, draw_all=True):
        mne.viz.plot_raw_psd(self.raw, fmin=0.25, fmax=40, tmin=self.tmin, tmax=self.tmax, n_fft=8 * 256, ax=ax,
                             picks=None if draw_all else [self.get_ch_name(), self.get_chs()[15]])


class Part:
    bands = [[1.5, 3.75], [4, 7.75], [8, 10.25], [10.5, 13.25], [13.5, 18], [18.5, 30]]

    raw, ch = None, None
    tmin, tmax = None, None
    params = []

    def __init__(self, raw, ch, tmin, tmax, params=None):
        self.raw, self.ch = raw, ch
        self.tmin, self.tmax = tmin, tmax

        if params is None:
            t_idx = raw.time_as_index([tmin, tmax])
            signal, times = raw[ch, t_idx[0]:t_idx[1]]
            me, sd = np.mean(signal), np.std(signal)

            band_powers = []
            for band in self.bands:
                psd, x = psd_welch(raw, fmin=band[0], fmax=band[1], tmin=tmin, tmax=tmax, n_fft=8 * 256,
                                   n_overlap=4 * 256)
                power = np.trapz(psd[ch], x)
                band_powers.append(power)

            self.params = [me, sd] + band_powers
        else:
            self.params = params

    def get_norm_params(self):
        np_params = np.array(self.params)
        return (np_params[0:2] / 1e-5).tolist() + (np_params[2:8] / 1e-11).tolist()

    def get_params(self):
        return self.params

    def get_stat_params(self):
        return self.params[0:2]

    def get_freq_params(self):
        return self.params[2:8]


def calculate_moments(f, x, y):
    f_sum = np.sum(f)

    m1_x = np.sum([f[i] / f_sum * x[i] for i in range(len(f))])
    m1_y = np.sum([f[i] / f_sum * y[i] for i in range(len(f))])

    m234_x = np.sum([[f[i] / f_sum * np.power(x[i] - m1_x, p) for i in range(len(f))]
                     for p in [2, 3, 4]], 1)
    m234_y = np.sum([[f[i] / f_sum * np.power(y[i] - m1_y, p) for i in range(len(f))]
                     for p in [2, 3, 4]], 1)

    sd_x = np.sqrt(m234_x[0])
    sd_y = np.sqrt(m234_y[0])

    standard_ms_x = [m1_x, sd_x, m234_x[1] / sd_x ** 3, m234_x[2] / sd_x ** 4]
    standard_ms_y = [m1_y, sd_y, m234_y[1] / sd_y ** 3, m234_y[2] / sd_y ** 4]

    standard_ms = standard_ms_x + standard_ms_y

    return standard_ms