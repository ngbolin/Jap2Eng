import random

def read_file(main_filepath, 
              en_train_filepath, ja_train_filepath, 
              en_dev_filepath, ja_dev_filepath,
              separator='\t', N=2000000, split_prob=0.999):
    '''
    Reads and extract first N lines (default set at 2,000,000) of the JParaCrawl dataset stored in @file_path argument.

    Saves @split_prob*@N sentences to the training filepath: English sentences to @en_train_filepath, and Japanese sentences to @ja_train_filepath.
    Saves (1-@split_prob)*@N sentences to the dev filepath: English sentences to @en_dev_filepath, and Japanese sentences to @ja_dev_filepath.
    '''
    indices = list(range(N))
    random.shuffle(indices)
    
    train_count = int(N * split_prob)
    train_indices = set(indices[:train_count])

    # 2. Open all files at once using a ContextStack or multiple 'with'
    try:
        with open(main_filepath, 'r', encoding='utf-8') as f_in, \
             open(en_train_filepath, 'w', encoding='utf-8') as f_en_tr, \
             open(ja_train_filepath, 'w', encoding='utf-8') as f_ja_tr, \
             open(en_dev_filepath, 'w', encoding='utf-8') as f_en_dv, \
             open(ja_dev_filepath, 'w', encoding='utf-8') as f_ja_dv:

            for i in range(N):
                line = f_in.readline()
                if not line: break # Stop if file is shorter than N
                
                parts = line.strip().split(separator)
                if len(parts) < 4: continue # Skip malformed lines
                
                _, _, en, ja = parts

                # 3. Direct routing (O(1) lookup)
                if i in train_indices:
                    f_en_tr.write(en + '\n')
                    f_ja_tr.write(ja + '\n')
                else:
                    f_en_dv.write(en + '\n')
                    f_ja_dv.write(ja + '\n')
                    
    except FileNotFoundError:
        print("Error: The source file was not found.")

if __name__ == "__main__":
    main_filepath='../data/jpn-eng/JParaCrawl/en-ja.bicleaner05.txt'
    
    en_train_filepath='../data/jpn-eng/JParaCrawl/train.en'
    ja_train_filepath='../data/jpn-eng/JParaCrawl/train.ja'
    en_dev_filepath='../data/jpn-eng/JParaCrawl/dev.en'
    ja_dev_filepath='../data/jpn-eng/JParaCrawl/dev.ja'
    
    _ = read_file(main_filepath, 
              en_train_filepath, ja_train_filepath, 
              en_dev_filepath, ja_dev_filepath)