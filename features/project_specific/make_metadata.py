import json
import os
import pandas as pd
import os
from collections import defaultdict
from random import shuffle
import itertools
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help="""Directory containing classes, which
                contain slides, which contain tiles""")
parser.add_argument('--tile_dir', help="""String representing the intermediate
                directories between slide and tile""")
parser.add_argument('--val_split', default = None, help="""Percent of each class
                that should be in the validation set""")
parser.add_argument('--test_split', default = None, help="""Percent of each class
                that should be in the test set""")


def dir_to_list(data_dir, tile_dir):
    """
    Saves a list of tuples. (Path to jpeg, class, slide)

    data_dir: directory containing the classes directories
    tile_dir: string representing the intermediate classes between slide and jpeg
    """

    dir_list = [(os.path.join(data_dir, classes, slide, tile_dir, jpg), classes, slide)
        for classes in os.listdir(data_dir)
        for slide in os.listdir(os.path.join(data_dir, classes))
        if os.path.isdir(os.path.join(data_dir, classes, slide))
        for jpg in os.listdir(os.path.join(data_dir, classes, slide, tile_dir))
        if os.path.splitext(jpg)[1] == ".jpeg"]

    return dir_list


def split_classes(metadata_df, split_column, class_column, val_split, test_split):
    """
    splits a dataframe into train, val, and test using a specified column

    metadata_df: dataframe
    split_column: column to group dataframe by
    count_column: column to quantify the grouping by
    """

    # create a dict of lists, one list for each unique value in column
    df_list = defaultdict(list)
    classes = metadata_df[class_column].unique()

    for i in classes:
        pos = metadata_df[class_column] == i
        df_list[i] = metadata_df[pos]

    # split the data into train, val, and test by iterating over the dictionary
    # essentially splitting the data per class into train, val, and test

    out_df = list()

    for key in df_list:

        current_df = df_list[key]
        grouped_current_df = current_df.groupby([split_column]).count()
        grouped_current_df = grouped_current_df.sort_values([class_column], ascending = False)

        if grouped_current_df.shape[0] < 3:
            print("{} has too few slides".format(key))
            continue
        else:
            current_df['split'] = 'NA'

            val_counter = 0
            train_counter = 0
            val_goal = 0

            # first identify the validation set
            for index, row in grouped_current_df.iterrows():

                if val_goal <= val_counter:
                    current_key = row.name
                    current_df.loc[current_df[split_column] == row.name, 'split'] = "Train"
                    train_counter += row[class_column]

                elif val_goal > val_counter:

                    current_key = row.name
                    current_df.loc[current_df[split_column] == row.name, 'split'] = "Val"
                    val_counter += row[class_column]

                val_goal = val_split * (train_counter + val_counter)

            subset_df = current_df[current_df['split'] == "Train"]
            grouped_current_df = subset_df.groupby([split_column]).count()
            grouped_current_df = grouped_current_df.sort_values([class_column], ascending = False)

            test_counter = 0
            train_counter = 0
            test_goal = 0

            # then identify the test set
            for index, row in grouped_current_df.iterrows():

                if row['split'] == "Val":
                    continue

                elif test_goal <= test_counter:

                    current_key = row.name
                    current_df.loc[current_df[split_column] == row.name, 'split'] = "Train"
                    train_counter += row[class_column]

                elif test_goal > test_counter:

                    current_key = row.name
                    current_df.loc[current_df[split_column] == row.name, 'split'] = "Test"
                    test_counter += row[class_column]

                test_goal = test_split * (train_counter + test_counter)

        out_df.append([tuple(x) for x in current_df.values])

    save_list = list(itertools.chain.from_iterable(out_df))

    return save_list

def main():
    args = parser.parse_args()
    save_list = dir_to_list(args.data_dir, args.tile_dir)

    if args.val_split is not None and args.test_split is not None:

        try:
            (int(args.val_split) + int(args.test_split)) > 100
        except:
            raise Exception("val_split and test_split add to one")

        val_split = int(args.val_split) / 100
        test_split = int(args.test_split) / 100

        # make the list of tuples into a data frame
        dir_df = pd.DataFrame(save_list, columns =['Path', 'label', 'slide'])
        # add a column with the classes as numerical
        dir_df.label = pd.Categorical(dir_df.label)
        dir_df['label_num'] = dir_df.label.cat.codes

        save_list = split_classes(metadata_df = dir_df, split_column = 'slide',
        class_column = 'label', val_split = val_split, test_split = test_split)

        with open('metadata_split.txt', 'w') as filehandle:
            json.dump(save_list, filehandle)

    else:

        with open('metadata.txt', 'w') as filehandle:
            json.dump(save_list, filehandle)

if __name__ == "__main__":
    main()
