import io
import os

def write_file(fpath, content):
    """ Writes content to a file in fpath
    @param fpath (str): filepath to read data
    @param content (str): file content
    """
    f = open(fpath, 'w')
    f.write(content)
    f.close()
    return

def read_and_process_data(fpath):
    """ Reads the Japanese and English corpus and creates
    2 separate files (.ja and .en) for each train, dev and test dataset
    in the same data folder
    @param fpath (str): filepath to read data
    """

    # reads data
    f = io.open(fpath, mode="r", encoding="utf-8")
    d = f.read()
    
    # split on '\n' to get a list where each item is a Eng2Jap example
    # second split on '\t' to split each item into 2; first subitem is Eng translation, second subitem is Jap translation
    examples = [e.split('\t') for e in d.split('\n')]

    # creates 2 variables to hold english and japanese example
    # only look for examples where there is a 1 to 1 map (so length == 2)
    en_ex, ja_ex = [e[0] for e in examples if len(e) == 2], [e[1] for e in examples if len(e) == 2] 

    # join on '\n' to create the file
    opath_en = f'{fpath}.en'
    opath_ja = f'{fpath}.ja'
    
    write_file(opath_en, '\n'.join(en_ex))
    write_file(opath_ja, '\n'.join(ja_ex))

    os.remove(fpath)
    return 

read_and_process_data('ja_en_data/split/train')
read_and_process_data('ja_en_data/split/dev')
read_and_process_data('ja_en_data/split/test')