import tensorflow as tf

from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np

from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
from scipy import stats

import io
import os
import json
from PIL import Image
import cv2

import easygui

import Fragment
import ML


def ml_imgs(records_set):
    model = tf.keras.applications.resnet50.ResNet50(include_top=False,
                                                    input_shape=(300, 300, 3),
                                                    weights='imagenet', pooling='avg')

    model.summary()
    model.compile()

    img = records_set.get_record(0).get_fragment_tg_img(0, 4)

    plt.figure()
    plt.imshow(img)
    plt.show()

    print(img.size)

    img = img.convert('RGB')

    img_data = np.asarray(img)
    img_data = np.expand_dims(img_data, axis=0)
    predictions = model.predict(img_data, verbose=0)

    print(predictions.shape)


def ml_test(ml_data):
    data = np.array(ml_data)
    x = data[:, :35]
    y = data[:, 35]

    plt.figure()
    df = pd.DataFrame(x)
    corr = df.corr()
    sns.heatmap(corr)
    plt.show()

    pca = PCA(n_components=3)
    pc = pca.fit_transform(x)
    print(pca.explained_variance_ratio_)

    res = [[] for i in range(int(max(y)) + 1)]
    for i in range(len(y)):
        res[int(y[i])].append(pc[i])
    res = [np.array(res[i]) for i in range(len(res))]
    res = [pd.DataFrame(res[i]) for i in range(len(res))]
    res = [res[i][(np.abs(stats.zscore(res[i])) < 3).all(axis=1)] for i in range(len(res))]
    res = [res[i].sample(n=100) for i in range(len(res))]
    res = [res[i].to_numpy() for i in range(len(res))]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in range(len(res)):
        ax.scatter(res[i][:, 0], res[i][:, 1], res[i][:, 2])
        print(res[i].shape)
    plt.show()


class MenuPanel:
    records_set = None
    video_path, video_offsets = None, {'FoDo': 0, 'FzDo': 30}

    fig, gs = None, None

    choose_records_set_button, choose_records_set_button_ax = None, None
    show_psd_button, show_psd_button_ax = None, None
    show_tg_button, show_tg_button_ax = None, None
    create_dataset_button, create_dataset_button_ax = None, None

    def __init__(self):
        self.fig = plt.figure()
        self.gs = self.fig.add_gridspec(4, 3)

        self.choose_records_set_button_ax = self.fig.add_subplot(self.gs[0, :])
        self.choose_records_set_button = Button(self.choose_records_set_button_ax, 'Open files')
        self.choose_records_set_button.on_clicked(lambda _: self.choose_records_set())

        self.show_psd_button_ax = self.fig.add_subplot(self.gs[1, :])
        self.show_psd_button = Button(self.show_psd_button_ax, 'Show PSD')
        self.show_psd_button.on_clicked(lambda _: self.show_psd())

        self.show_tg_button_ax = self.fig.add_subplot(self.gs[2, :])
        self.show_tg_button = Button(self.show_tg_button_ax, 'Show TG')
        self.show_tg_button.on_clicked(lambda _: self.show_tg())

        self.create_dataset_button_ax = self.fig.add_subplot(self.gs[3, :])
        self.create_dataset_button = Button(self.create_dataset_button_ax, 'Save dataset')
        self.create_dataset_button.on_clicked(lambda _: self.create_dataset())

        plt.show()

    def choose_records_set(self):
        input_folder = easygui.diropenbox('Choose EEG records', 'Open files')
        self.records_set = Fragment.EEGRecordsSet(input_folder)

        filenames = os.listdir(input_folder)
        for filename in filenames:
            if filename.endswith('.mp4'):
                self.video_path = os.path.join(input_folder, filename)

        with open(os.path.join(input_folder, 'Video_layout.json')) as json_file:
            self.video_offsets = json.load(json_file)

        easygui.msgbox("Success")

    def show_psd(self):
        psd_panel = PSDPanel(self.records_set)

        psd_panel.show()

    def show_tg(self):
        main_panel = MainTGPanel(self.records_set, self.video_path, self.video_offsets)

        band = easygui.indexbox('Choose band', 'Frequency band', ['Me', 'Sd', 'De', 'Th', 'A1', 'A2', 'B1', 'B2'])
        ch = easygui.indexbox('Choose channel', 'EEG channel', [
            "Fp1", "Fpz", "Fp2", "F7", "F3", "Fz",
            "F4", "F8", "FT7", "FC3", "FCz", "FC4", "FT8",
            "T3", "C3", "Cz", "C4", "T4", "TP7",
            "CP3", "CPz", "CP4", "TP8", "T5",
            "P3", "Pz", "P4", "T6", "O1", "Oz", "O2"
        ])

        main_panel.show(band, ch)

    def create_dataset(self):
        self.test()

    def ml_scheme_1(self):
        ml_data = self.records_set.get_ml_data()
        filepath = easygui.filesavebox('Save data', 'Create dataset')
        np.savetxt(filepath, ml_data, delimiter=',')

    def ml_scheme_2(self):
        imgs, df = self.records_set.get_ml_tg_imgs()

        dirpath = easygui.diropenbox('Save data', 'Create dataset')
        imgs_dirpath = os.path.join(dirpath, 'images')
        if not os.path.exists(imgs_dirpath):
            os.mkdir(imgs_dirpath)

        print(df)
        for i in range(len(imgs)):
            print(df.loc[i].label_group)
            path = os.path.join(imgs_dirpath, df.loc[i].file_path)
            imgs[i].save(path)

        df_path = os.path.join(dirpath, 'df.csv')
        df.to_csv(df_path, mode='a', header=not os.path.exists(df_path), index=False)

    def test(self):
        ML.train_model()


class PSDPanel:
    records_set = None

    drop_button, drop_button_ax = None, None

    def __init__(self, records_set):
        self.records_set = records_set

    def show(self):
        fig, axes = plt.subplots(self.records_set.get_records_number() + 1, 1)

        for i in range(self.records_set.get_records_number()):
            record = self.records_set.get_record(i)
            axes[i].set_xlabel(record.get_record_name())
            record.draw_psd(axes[i])

        self.drop_button_ax = axes[-1]
        self.drop_button = Button(self.drop_button_ax, 'Drop channel')
        self.drop_button.on_clicked(lambda _: self.drop_ch())

        plt.get_current_fig_manager().window.state("zoomed")
        plt.show()

    def drop_ch(self):
        ch = easygui.indexbox('Choose channel', 'EEG channel', self.records_set.get_chs())
        records_names = easygui.multchoicebox('Choose record', 'EEG records', self.records_set.get_records_names(),
                                              preselect=None)

        for record_name in records_names:
            record = self.records_set.get_record_by_name(record_name)
            record.hide_ch(ch)
            fragments = record.get_fragments_by_ch(ch)
            for fragment in fragments:
                fragment.set_status(False)


class MainTGPanel:
    records_set = None
    video_path, video_offsets = None, {}

    buttons = []

    def __init__(self, records_set, video_path, video_offsets):
        self.records_set = records_set
        self.video_path, self.video_offsets = video_path, video_offsets

    def show(self, band, ch):

        def fragment_clicked(record_idx, fragment_idx):
            records_set = self.records_set
            record = records_set.get_record(record_idx)
            fragment = record.get_fragment(ch, fragment_idx)
            return lambda _: FragmentPanel(fragment, self.video_path,
                                           self.video_offsets[fragment.get_record_name()]).show()

        imgs, labels = self.records_set.get_tg_imgs(band)

        fig, axes = plt.subplots(self.records_set.get_records_number(), self.records_set.get_max_fragments_number())
        for i in range(len(axes)):
            for j in range(len(axes[i])):
                # axes[i][j].axis('off')
                if j < len(imgs[i]):
                    axes[i][j].set_xlabel(labels[i][j])

                    tg_button = Button(axes[i][j], '', image=imgs[i][j])
                    tg_button.on_clicked(fragment_clicked(i, j))
                    self.buttons.append(tg_button)
                else:
                    axes[i][j].axis('off')

        plt.get_current_fig_manager().window.state("zoomed")
        plt.show()


class FragmentPanel:
    fragment = None
    video_path, video_offset = None, 0

    fig, gs = None, None
    parts_ax, table_ax, psd_ax, video_ax = None, None, None, None

    choose_button, choose_button_ax = None, None
    drop_button, drop_button_ax = None, None
    drop_all_button, drop_all_button_ax = None, None
    time_slider, time_slider_ax = None, None

    video, cur_img = None, None

    def __init__(self, fragment, video_path, video_offset):
        self.fragment = fragment
        self.video_path, self.video_offset = video_path, video_offset

        self.fig = plt.figure()
        self.gs = self.fig.add_gridspec(20, 30)

        self.parts_ax = self.fig.add_subplot(self.gs[0:10, 11:20], polar='True')
        self.table_ax = self.fig.add_subplot(self.gs[1:10, 21:])
        self.psd_ax = self.fig.add_subplot(self.gs[12:, 12:])
        self.video_ax = self.fig.add_subplot(self.gs[:19, 0:10])

        self.choose_button_ax = self.fig.add_subplot(self.gs[0:1, 21:23])
        self.drop_button_ax = self.fig.add_subplot(self.gs[0:1, 23:25])
        self.drop_all_button_ax = self.fig.add_subplot(self.gs[0:1, 25:27])
        self.time_slider_ax = self.fig.add_subplot(self.gs[19, 0:10])

        self.draw_parts()
        self.draw_table()
        self.draw_psd()
        self.draw_video()

        self.create_buttons()

    def draw_parts(self):
        labels = ['Me', 'Sd', 'De', 'Th', 'A1', 'A2', 'B1', 'B2']
        ax = self.parts_ax

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        angles_circuit = list(angles) + [angles[0]]
        for part in self.fragment.get_norm_parts_data():
            data_circuit = part + [part[0]]
            ax.plot(angles_circuit, data_circuit, 'o-', linewidth=2)
            ax.fill(angles_circuit, data_circuit, alpha=0.25)
        ax.set_thetagrids(angles * 180 / np.pi, labels)

    def draw_table(self):
        col_labels = ['Me', 'Sd', 'De', 'Th', 'A1', 'A2', 'B1', 'B2']
        row_labels = ['P1', 'P2', 'P3', 'P4', 'AG', 'RG']
        ax = self.table_ax
        ax.axis('off')

        data = self.fragment.get_norm_parts_data()
        data.append(self.fragment.get_norm_parts_avg())
        data.append(self.fragment.get_norm_parts_rng())
        table_data = [[float("{:.2f}".format(p)) for p in part] for part in data]

        table = ax.table(cellText=table_data, rowLabels=row_labels, colLabels=col_labels, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.15, 1.15)

    def draw_psd(self):
        ax = self.psd_ax
        self.fragment.draw_psd(ax)

    def draw_video(self):
        ax = self.video_ax
        ax.axis('off')

        self.video = cv2.VideoCapture(self.video_path)
        time = self.fragment.get_tmin()

        self.update_video(time)

    def update_video(self, time):
        ax = self.video_ax

        fps = self.video.get(cv2.CAP_PROP_FPS)
        frame_index = fps * (time + self.video_offset)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        res, frame = self.video.read()
        if res:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.cur_img is not None:
                self.cur_img.set(data=img)
                ax.relim()
            else:
                self.cur_img = ax.imshow(img)

    def create_buttons(self):
        self.choose_button = Button(self.choose_button_ax, 'Choose')
        self.choose_button.on_clicked(lambda _: self.choose_ch())

        self.drop_button = Button(self.drop_button_ax, 'Drop')
        self.drop_button.on_clicked(lambda _: self.drop_chs(self.fragment))

        self.drop_all_button = Button(self.drop_all_button_ax, 'Drop all')
        self.drop_all_button.on_clicked(lambda _: self.drop_chs())

        self.time_slider = Slider(self.time_slider_ax, 'Time', self.fragment.get_tmin(), self.fragment.get_tmax())
        self.time_slider.on_changed(lambda _: self.update_video(self.time_slider.val))

    def choose_ch(self):
        choice = easygui.indexbox('Choose channel', 'EEG channel', self.fragment.get_raw().ch_names)
        new_fragment = self.fragment.get_other_chs()[choice]
        FragmentPanel(new_fragment, self.video_path, self.video_offset).show()
        plt.close(self.fig)

    def drop_chs(self, ch=None):
        if ch is None:
            for ch in self.fragment.get_other_chs():
                ch.set_status(False)
        else:
            ch.set_status(False)

    def show(self):
        plt.get_current_fig_manager().window.state("zoomed")
        plt.show()


class PanelImg:
    fig = None

    def __init__(self, fig):
        self.fig = fig

    def draw(self):
        matplotlib.use('Agg')

        buf = io.BytesIO()
        self.fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        plt.close(self.fig)
        return img
