import os
import music21 as m21
import json
from tensorflow import keras
import numpy as np



dataset_path = "deutschl/erk"
save_dir = "dataset"
SINGLE_FILE_DATASET = "file_dataset"
MAPPING_PATH = "mapping.json"
SEQUENCE_LENGTH = 64

acceptable_durations = [0.25,   #十六分音符
                        0.5,
                        0.75,   #附点八分音符
                        1.0,    #四分音符
                        1.5,     #附点四分音符
                        2,
                        3,
                        4
                        ]

def load_songs_in_krn(dataset_path):
    songs = []

    # go through all the files in dataset and load them with music21
    for path, subdir, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "krn":
                """使用music21库加载数据集(文件路径python字符串)并将他们转化为stream"""
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs

def has_acceptable_durations(song, acceptable_durations):             #布尔函数
    for note in song.flat.notesAndRests:    #选择song中的音符属性(.flat将song的所愿属性展开 .notesAndRests留下音符属性）
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True

def transpose(song):

    # get key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]
    # return key 这里是个测试

    # estimate(估计) key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # get interval(间隔) for transposition. E.g. Bmaj ->Cmaj
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
        #在和声分析中，音阶内以主音tonic（音阶的第一个音，也是音阶中最重要的音）为I级
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # transpose song by calculated interval
    transpose_song = song.transpose(interval)

    return transpose_song

def encode_song(song, time_step=0.25):
    # pitch=60，duration=1.0  -> [60, "_", "_", "_"]

    encoded_song = []

    for event in song.flat.notesAndRests:

        # handel notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        # convert the note/rest into time series notation(用时间序列符号表示)
        steps = int(event.duration.quarterLength / time_step)   # 1.0 ->4 代表四分音符
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
    # cast encoded song to str
    encoded_song = " ".join(map(str, encoded_song))   #直接join会报错，要先全部映射为字符串
    # print(encode_song)测试

    return encoded_song


def preprocess(dataset_path):


    # load the folk songs
    print("loading songs...")
    songs = load_songs_in_krn(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    # filter out songs that have non-acceptable durations(不同拍号)
    for i, song in enumerate(songs):
        if not has_acceptable_durations(song, acceptable_durations):
            continue                # 如果不在范围内将跳过它，不对它进行处理


        # transpose songs to Cmaj/Amin
        song = transpose(song)

        # encode songs with music time series representation
        encoded_song = encode_song(song)

        # save songs to text file
        save_path = os.path.join(save_dir, str(i+1))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)

def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song

def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter = "/ " * sequence_length  # 创建与序列长度相同的定界符数为了使网络可以得知我们即将结束这首歌的旋律
    songs = ""

    # load encoded songs and delimiters(定界符)
    for path, _, files in os.walk(dataset_path):   # 这里代码有点小问题
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter
    songs = songs[:-1]      # 去除结尾的空格

    # save the string that contains all dataset
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)
    return songs

def create_mapping(songs, mapping_path):
    mappings = {}
    # indetify the vocabularly
    songs = songs.split()  # 分割字符串中的空格返回数组
    vocabulary = list(set(songs))     # set()转化为集合(集合类型具有不重复性)去除重复的元素，然后再转化为列表

    # create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i


    # save vocabulary to a jason file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)  # 缩进为4

def convert_songs_to_int(songs):
    int_songs = []

    # load mappings
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    # cast songs string to a list
    songs = songs.split()

    # map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs

def generate_training_sequences(sequence_length):
    # [11, 12, 13, 14 ...] -> input: [11, 12], target: 13

    # load songs and map them to int
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    # generate the training sequences
    # 100 symbols, 64 sequence_length, number of sequence = 100 - 64 = 36
    inputs = []
    targets = []

    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):                              # 在序列中向右迭代
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])
        #print(np.array(targets).shape)
        #print(targets)
    # one-hot encode the sequences
    # inputs: (number of sequences, sequence length)
    vocabulary_size = len(set(int_songs))     # 创建独热编码字典
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)   # 将原始向量变为独热编码
    targets = np.array(targets)             # 转换为numpy数组

    return inputs, targets





def main():
    preprocess(dataset_path)
    songs = create_single_file_dataset(save_dir, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)



if __name__ == "__main__":
    main()

    # songs = load_songs_in_krn(dataset_path)
    # print(f"Loaded {len(songs)} songs.")
    # for index, i in enumerate(songs):
    #
    #     if index ==0 :
    #         print(i)
    #         print(transpose(i))
    #         a = transpose(i)
    #         i.show()
    #         a.show()
    #         a = encode_song(a)
    #         print(a)
    # preprocess(dataset_path)
    # songs = create_single_file_dataset(save_dir, SINGLE_FILE_DATASET,SEQUENCE_LENGTH)
    # create_mapping(songs=songs)
    #song = songs[0]
    #song = song.write('midi',fp='test.midi')
    #song.show()
    # f = m21.note.Note('C4')
    # print(f.duration)
    #f.show()
