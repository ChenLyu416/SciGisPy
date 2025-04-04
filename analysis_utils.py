import json
import evaluate
import os
import shutil
import pathlib
import nltk
import numpy as np
import pandas as pd
import stanza
import subprocess
import matplotlib.pyplot as plt
import readability
from readability.readability import Readability

sari = None
rouge = None
bleu = None


def get_dataset_descr_by_name(name, datasets):
    return next((dataset for dataset in datasets if dataset['name'] == name))


def get_gis_config_by_name(name, gis_configs):
    return next((config for config in gis_configs if config['name'] == name))


def gis_save_name(dataset_descr, key_simplified='gen'):
    key_n = '' if key_simplified == 'gen' else f'_{key_simplified}'
    return f"{dataset_descr['name']}{key_n}_gist"


def parse_dataset(filename):
    with open(filename, 'r') as file:
        dataset_unfiltered = json.load(file)

    try:
        for entry in dataset_unfiltered:
            entry['doi'] = entry['doi'].replace('/', '-').replace(':', '-')
    except Exception:
        print('Exception occured when dealing with doi.')

    return dataset_unfiltered


def dataset_filtered_min_sentence_count(dataset_unfiltered, min_sentence_count, keys=['abstract', 'output']):
    def long_enough(entry):
        for key in list(entry.keys())[1:]:
            if entry[key].count('.') <= min_sentence_count:
                print(entry[key])
                return False
        return True

    return list(filter(long_enough, dataset_unfiltered))


def set_up_prerequisites():
    stanza.install_corenlp()
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('wordnet_ic')


def init_metrics():
    global sari, rouge, bleu
    sari = evaluate.load('sari')
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')


def score_gist(id_data_iter, gis_config_file, save_name, overwrite_save=False, result_column='gis'):
    docs_path = 'GisPy/data/documents'
    gispy_path = 'GisPy/gispy'
    saves_dir = 'saves'
    gis_config_target_file = 'gis_config.json'
    scores_file = save_name + '.csv'

    n_entries = 0
    ids_list = []
    data_list = []
    for entry in id_data_iter:
        n_entries += 1
        ids_list.append(entry[0])
        data_list.append(entry[1])

    if overwrite_save or not os.path.exists(os.path.join(gispy_path, saves_dir, scores_file)):
        pathlib.Path(docs_path).mkdir(exist_ok=True)
        del_list = os.listdir(docs_path)

        for f in del_list:
            os.remove(os.path.join(docs_path, f))

        for i in range(n_entries):
            with open(os.path.join(docs_path, ids_list[i]) + '.txt', 'w', encoding="utf-8", errors='strict') as f:
                f.write(data_list[i])

        shutil.copyfile(os.path.join(gispy_path, gis_config_file), os.path.join(gispy_path, gis_config_target_file))
        
        proc = subprocess.Popen(['python3', 'run.py', os.path.join(saves_dir, scores_file)], cwd=gispy_path,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        while True:
            output = proc.stdout.readline()
            if proc.poll() is not None:
                break

            if output:
                print(output)

        if proc.returncode != 0:
            raise ChildProcessError('Error: GisPy finished with an error.')

        print('GisPy has finished, extracting results...')

    else:
        print('Existing save found, extracting results...')

    table = pd.read_csv(os.path.join(gispy_path, saves_dir, scores_file), header=0)
    
    # results = np.empty((len(table),), dtype='float32')

    # for i, id in enumerate(ids_list):
    #     d_id = id + '.txt'
    #     try:
    #         results[i] = table[table['d_id'] == d_id][result_column]
    #     except Exception:
    #         print('Exception occured while enumerate GIS result for data' + d_id)
    #         results[i] = -999
    
    # return results

    return table[['d_id', result_column]].to_numpy()


def score_gist_multiple_versions(num_versions, id_doc_tuple_iter, gis_config, save_name, overwrite_save=False,
                                 result_column='gis'):
    def id_data_iter(id_doc_tuple_iter):
        for id, doc_tuple in id_doc_tuple_iter:
            for t_index, doc in enumerate(doc_tuple):
                yield f'{id}-v{t_index}', doc

    all_gist = score_gist(id_data_iter(id_doc_tuple_iter), gis_config['file'], f"{save_name}-{gis_config['name']}",
                          overwrite_save, result_column=result_column)
    gist_scores = []

    for _ in range(num_versions):
        gist_scores.append(np.empty((int(len(all_gist) / num_versions),), dtype='float32'))
    
    for i, gist in enumerate(all_gist):
        current_list = gist_scores[i % num_versions]

        current_list[i // num_versions] = gist[1]
    
    return gist_scores


def plot_document_scores(x_score, y_score, x_label, y_label, plot_separation_line=False, print_correlation=True):
    plt.scatter(x_score, y_score)
    if plot_separation_line:
        line = np.array([max(np.min(x_score), np.min(y_score)), min(np.max(x_score), np.max(y_score))])
        plt.plot(line, line)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

    print(np.corrcoef(x_score, y_score)[0,1])


def plot_document_scores_multi_set(x_score_sets, y_score_sets, x_label, y_label, colours, plot_separation_line=False):

    all_x = np.empty((0,), dtype=x_score_sets[0].dtype)
    all_y = np.empty((0,), dtype=y_score_sets[0].dtype)
    for i in range(len(x_score_sets)):
        colour = colours[i]
        x_score = x_score_sets[i]
        y_score = y_score_sets[i]
        plt.scatter(x_score, y_score, c=colour)

        all_x = np.concatenate((all_x, x_score))
        all_y = np.concatenate((all_y, y_score))

    if plot_separation_line:
        line = np.array([max(np.min(all_x), np.min(all_y)), min(np.max(all_x), np.max(all_y))])
        plt.plot(line, line)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def score_sari(id_orig_pred_ref_iter, data_len):
    result = np.empty((data_len,), dtype='float32')
    for i, entry in enumerate(id_orig_pred_ref_iter):
        id, orig, pred, ref = entry
        score = sari.compute(sources=[orig], predictions=[pred], references=[[ref]])
        result[i] = score['sari']

    return result


def score_bleu(id_data_ref_iter, data_len):
    result = np.empty((data_len,), dtype='float32')
    for i, entry in enumerate(id_data_ref_iter):
        id, data, ref = entry
        score = bleu.compute(predictions=[data], references=[ref])
        result[i] = score['bleu']

    return result


def score_rouge(id_data_ref_iter, data_len):
    result = np.empty((data_len, 2), dtype='float32')
    for i, entry in enumerate(id_data_ref_iter):
        id, data, ref = entry
        score = rouge.compute(predictions=[data], references=[ref])
        result[i,0] = score['rouge1']
        result[i,1] = score['rouge2']

    return result


# adapted from https://studymachinelearning.com/jaccard-similarity-text-similarity-metric-in-nlp/
def jaccard_similarity(doc1, doc2):

    words_doc1 = set(doc1.lower().split())
    words_doc2 = set(doc2.lower().split())

    intersection = words_doc1.intersection(words_doc2)
    union = words_doc1.union(words_doc2)

    return float(len(intersection)) / len(union)

def score_jaccard_similarity(id_orig_pred_iter, data_len):
    result = np.empty((data_len,), dtype='float32')
    for i, entry in enumerate(id_orig_pred_iter):
        id, orig, pred = entry
        result[i] = jaccard_similarity(orig, pred)

    return result


def score_readability(id_data_iter, data_len):
    result = np.empty((data_len,2), dtype='float32')
    for i, entry in enumerate(id_data_iter):
        id, data = entry

        readability = Readability(data)
        result[i,0] = readability.ARI()
        result[i,0] = readability.FleschKincaidGradeLevel()

    return result


def filter_dataset_ari_len(dataset):
    dataset_ari_len = filter(lambda entry: len(entry['pls'].split()) >= 100 and
                                           len(entry['abstract'].split()) >= 100 and
                                           len(entry['gen'].split()) >= 100, dataset)
    return list(dataset_ari_len)


def test_gis_diff_for_configs(dataset_descr, configs, min_sentence_count, quiet=False, add_blank_rows=False,
                              add_dataset_name=False, key_simplified='gen'):

    dataset_unfiltered = parse_dataset(dataset_descr['file'])
    dataset_filtered_all = dataset_filtered_min_sentence_count(dataset_unfiltered, min_sentence_count)

    abstracts_gist_unf_per_config = []
    gen_gist_unf_per_config = []

    col_config = 'Config'
    col_min_sent = 'Sentence count min.'
    col_doc_count = '# documents'
    col_mean_diff = 'μ GIS diff'
    col_percent_pos_diff = '% pos. GIS diff'
    col_pos_diff = '# pos. GIS diff'
    col_non_pos_diff = '# non-pos. GIS diff'

    table = pd.DataFrame(columns=[col_config, col_min_sent, col_doc_count, col_mean_diff, col_percent_pos_diff,
                                  col_pos_diff, col_non_pos_diff])

    if add_dataset_name:
        table = table.append({col_config: dataset_descr['name']}, ignore_index=True)
        table = table.append({col_config: ' '}, ignore_index=True)

    for config in configs:
        gist_all = score_gist_multiple_versions(2, map(lambda entry: (entry['doi'],
                                                                      (entry['abstract'], entry[key_simplified])),
                                                       dataset_filtered_all),
                                                config, gis_save_name(dataset_descr, key_simplified=key_simplified))

        abstracts_gist_filtered = gist_all[0]
        gen_gist_filtered = gist_all[1]

        abstracts_gist_unf_per_config.append(abstracts_gist_filtered)
        gen_gist_unf_per_config.append(gen_gist_filtered)

        if not quiet:
            print()
            print(f"***** For config {config['name']}: *****")
            print()

        dataset = dataset_filtered_all

        filtered_indices = np.empty((len(dataset),), dtype='uint16')
        for i, entry in enumerate(dataset):
            filtered_indices[i] = next(i for i, v in enumerate(dataset_unfiltered) if v is entry)

        abstracts_gist = abstracts_gist_filtered
        gen_gist = gen_gist_filtered

        gist_diff = gen_gist - abstracts_gist
        if not quiet:
            print(f'** Min. {min_sentence_count} estimated sentences: **')
            print(f'Number of document pairs: {len(dataset)}')

            print(f'Mean GIS diff: {np.mean(gist_diff)}')
            print(f'% positive GIS diff: {100.0 * len(gist_diff[gist_diff > 0.0]) / len(dataset)}')
            print(f'Positive GIS diff: {len(gist_diff[gist_diff > 0.0])}')
            print(f'Non-positive GIS diff: {len(gist_diff[gist_diff <= 0.0])}')
            print()

            plot_document_scores(abstracts_gist, gen_gist, 'GIS - original', 'GIS - simplified',
                                    plot_separation_line=True)
            print()

        entry = {
            col_config: config['name'],
            col_min_sent: min_sentence_count,
            col_doc_count: len(dataset),
            col_mean_diff: np.mean(gist_diff),
            col_percent_pos_diff: 100.0 * len(gist_diff[gist_diff > 0.0]) / len(dataset),
            col_pos_diff: len(gist_diff[gist_diff > 0.0]),
            col_non_pos_diff: len(gist_diff[gist_diff <= 0.0])
        }
        table = table.append(entry, ignore_index=True)

        if add_blank_rows:
            table = table.append({col_config: ' '}, ignore_index=True)

    return table

