import os
import re
import json
import numbers
import itertools
import statistics
import time
import math
import torch

import pandas as pd
import numpy as np

from os import listdir
import scipy.stats as stats
from os.path import isfile, join
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from stanza.server import CoreNLPClient
from sentence_transformers import SentenceTransformer, util

from utils import find_mrc_word
from utils import get_causal_cues
from utils import read_megahr_concreteness_imageability
from data_reader import GisPyData

import sent2vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from scipy.spatial import distance

import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# import openai
# from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from semantic_chunking_pack.llama_index.packs.node_parser_semantic_chunking.base import SemanticChunker


class GIST:
    def __init__(self, gispy_config_path='gispy_config.json', megahr_path='../data/megahr/megahr.en.sort.i'):
        print('loading parameters and models...')

        # loading parameters
        config_path = gispy_config_path
        if os.path.exists(config_path):
            with open(config_path) as f:
                self.params = json.load(f)
        else:
            raise FileNotFoundError('Please put the config file in the following path: /gist_config.json')
        
        if os.path.exists("SEMCHUNK_config.json"):
            with open("SEMCHUNK_config.json") as f:
                self.SEMCHUNK_params = json.load(f)
        else:
            self.SEMCHUNK_params = {
                "buffer": 1,
                "percentile": 90
            }

        # reading megahr
        self.megahr_dict = read_megahr_concreteness_imageability(megahr_path)
        self.verb_similarity = dict()
        self.sentence_model = None

        self.LLM_sentence_tokenizer = None
        self.LLM_sentence_model = None

        # compile the causal patterns
        causal_cues = get_causal_cues()
        self.causal_patterns = list()
        for idx, row in causal_cues.iterrows():
            self.causal_patterns.append(re.compile(r'' + row['cue_regex'].lower() + ''))

        self.ic = wordnet_ic.ic('ic-brown.dat')
        self.max_ic_counts = {
            wn.VERB: max(self.ic[wn.VERB].values()),
            wn.NOUN: max(self.ic[wn.NOUN].values())
        }
        
        self.mean_ic_counts = {
            wn.VERB: sum(self.ic[wn.VERB].values()) / len(self.ic[wn.VERB].values()),
            wn.NOUN: sum(self.ic[wn.NOUN].values()) / len(self.ic[wn.NOUN].values())
        }

    @staticmethod
    def _local_cosine(embeddings):
        """
        computing cosine only for consecutive sentence embeddings
        :param embeddings:
        :return:
        """
        if len(embeddings) <= 1:
            return 0
        else:
            i = 0
            scores = list()
            while i + 1 < len(embeddings):
                scores.append(util.cos_sim(embeddings[i], embeddings[i + 1]).item())
                i += 1
            return statistics.mean(scores)

    @staticmethod
    def _global_cosine(embeddings):
        """
        computing cosine of all pairs of sentence embeddings
        :param embeddings:
        :return:
        """
        scores = list()
        pairs = itertools.combinations(embeddings, r=2)
        for pair in pairs:
            scores.append((util.cos_sim(pair[0], pair[1]).item()))
        return statistics.mean(scores) if len(scores) > 0 else 0

    @staticmethod
    def _clean_text(text):
        encoded_text = text.encode("ascii", "ignore")
        text = encoded_text.decode()
        # text = text.replace('…', '...')
        text = text.replace('…', ' ')
        # this doesn't make sense, why remove all singular new-line characters?
        # text = re.sub(r'(?<!\n)\n(?!\n)', '', text)
        text = re.sub(' +', ' ', text)
        text = re.sub(r'\n+', '\n', text).strip()
        return text

    def compute_indices(self, documents, gis_config):
        """
        computing the Gist Inference Score (GIS) for a collection of documents
        :return:
        """
        df_cols = ["d_id", "text"]
        gispy_data = GisPyData()
        df_cols.extend(gispy_data.get_gispy_index_columns())
        df_docs = pd.DataFrame(columns=df_cols)
        errors_log = list()

        class MockNLPClient(object):
            def __enter__(self):
                pass

            def __exit__(self, *args):
                pass

        if self._should_compute_any(gis_config, ['CoREF']):
            client_ = CoreNLPClient(
                annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'coref'],
                threads=10,
                timeout=60000,
                memory='20G',
                be_quiet=True)
        else:
            client_ = MockNLPClient()

        with client_ as client:
            # chose sentence embedding model for PCREF
            if self._should_compute_any(gis_config, ['sen_model_mxbai']):
                self.sentence_model = SentenceTransformer(self.params['sentence_transformers_model_mxbai'], device = "cuda:1")
                print('mxbai sentence model for PCREF loaded')
            elif self._should_compute_any(gis_config, ['sen_model_cse']):
                self.sentence_model = SentenceTransformer(self.params['sentence_transformers_model_cse'], device = "cuda")
                print('cse sentence model for PCREF loaded')
            elif self._should_compute_any(gis_config, ['sen_model_biobert']):
                self.sentence_model = SentenceTransformer(self.params['sentence_transformers_model_biobert'], device = "cuda")
                print('biobert sentence model for PCREF loaded')
            elif self._should_compute_any(gis_config, ['sen_model_mistral']):
                print('Loading tokenizer for Mistral model...')
                self.LLM_sentence_tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct', torch_dtype=torch.float16)
                print('Loading Mistral model...')
                self.LLM_sentence_model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct', torch_dtype=torch.float16, device_map="auto")
                print('Mistral sentence model for PCREF loaded')
            elif self._should_compute_any(gis_config, ['sen_model_biosentvec']):
                print('Loading BioSentVec sentence model...')
                model_path = "BioSentVec_PubMed_MIMICIII-bigram_d700.bin"
                self.sentence_model = sent2vec.Sent2vecModel()
                try:
                    self.sentence_model.load_model(model_path)
                except Exception as e:
                    print(e)
                print('BioSentVec model for PCREF loaded')
            else:
                print('Loading default sentence model...')
                self.sentence_model = SentenceTransformer(self.params['sentence_transformers_model'])  


            if self._should_compute_any(gis_config, ['SEMCHUNK_mxbai']):
                print('Loading mxbai model for SEMCHUNK...')
                self.semchunk_embed_model = HuggingFaceEmbedding(model_name='mixedbread-ai/mxbai-embed-large-v1', device = "cuda:1")
                self.semchunk_splitter = SemanticChunker(buffer_size=self.SEMCHUNK_params['buffer'], breakpoint_percentile_threshold=self.SEMCHUNK_params['percentile'], embed_model=self.semchunk_embed_model)
            elif self._should_compute_any(gis_config, ['SEMCHUNK_cse']):
                print('Loading cse model for SEMCHUNK...')
                self.semchunk_embed_model = HuggingFaceEmbedding(model_name='kamalkraj/BioSimCSE-BioLinkBERT-BASE')
                self.semchunk_splitter = SemanticChunker(buffer_size=self.SEMCHUNK_params['buffer'], breakpoint_percentile_threshold=self.SEMCHUNK_params['percentile'], embed_model=self.semchunk_embed_model)
            elif self._should_compute_any(gis_config, ['SEMCHUNK_biobert']):
                print('Loading biobert model for SEMCHUNK...')
                self.semchunk_embed_model = HuggingFaceEmbedding(model_name='gsarti/biobert-nli')
                self.semchunk_splitter = SemanticChunker(buffer_size=self.SEMCHUNK_params['buffer'], breakpoint_percentile_threshold=self.SEMCHUNK_params['percentile'], embed_model=self.semchunk_embed_model)
            elif self._should_compute_any(gis_config, ['SEMCHUNK_mistral']):
                if self.LLM_sentence_tokenizer is None:
                    print('Loading Mistral tokenizer for SEMCHUNK...')
                    self.LLM_sentence_tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral', torch_dtype=torch.float16)
                    print('Mistral tokenizer for SEMCHUNK loaded')
                if self.LLM_sentence_model is None:
                    print('Loading Mistral model for SEMCHUNK...')
                    self.LLM_sentence_model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Mistral', torch_dtype=torch.float16, device_map="auto")
                    print('Mistral model for SEMCHUNK loaded')
                print('Intializing semantic chunker...')
                self.semchunk_splitter = SemanticChunker(buffer_size=self.SEMCHUNK_params['buffer'], breakpoint_percentile_threshold=self.SEMCHUNK_params['percentile'], 
                                                         embed_tokenizer=self.LLM_sentence_tokenizer, use_sentence_transformer=False)
                print('Splitter initialized')
            
        

            for i, doc in enumerate(documents):
                txt_file, doc_text = doc
                ttt = time.time()
                doc_text = self._clean_text(doc_text)

                bool_f = self._should_compute_any(gis_config, ['SMCAUSf_1', 'SMCAUSf_a', 'SMCAUSf_1p', 'SMCAUSf_ap'])
                bool_b = self._should_compute_any(gis_config, ['SMCAUSb_1', 'SMCAUSb_a', 'SMCAUSb_1p', 'SMCAUSb_ap'])
                df_doc, token_embeddings = gispy_data.convert_doc(doc_text, use_fasttext=bool_f, use_biowordvec=bool_b)

                doc_sentences, n_paragraphs, n_sentences = self._get_doc_sentences(df_doc)
                token_ids_by_sentence = self._get_doc_token_ids_by_sentence(df_doc)
                # -------------------------------
                # computing the coref using corenlp
                if self._should_compute_any(gis_config, ['CoREF']):
                    print('{}: Computing CoREF for #{}'.format(time.time() - ttt, i + 1))
                    try:
                        coref_scores = list()
                        for p_id, p_sentences in doc_sentences.items():
                            paragraph_text = ' '.join(p_sentences)
                            ann = client.annotate(paragraph_text)
                            chain_count = len(list(ann.corefChain))
                            coref_score = chain_count / len(p_sentences)
                            coref_scores.append(coref_score)
                        CoREF = statistics.mean(coref_scores)
                    except Exception as e:
                        CoREF = 0
                        errors_log.append('file: {}, message: {}'.format(txt_file, str(e)))
                        print('Computing CoREF failed. Message: {}'.format(str(e)))
                else:
                    CoREF = 0

                   
                sen_model_mxbai, sen_model_cse, sen_model_biobert, sen_model_mistral, sen_model_biosentvec = (0, 0, 0, 0, 0)

                PCREF1, PCREFa, PCREF1p, PCREFap = (0, 0, 0, 0)
                if self._should_compute_any(gis_config, ['PCREF_1', 'PCREF_a', 'PCREF_1p', 'PCREF_ap']):
                    sentence_embeddings = dict()
                    all_sentences = list()
                    processed_all_sentences = []
                    if self._should_compute_any(gis_config, ['sen_model_biosentvec']):
                        
                        # initializing sentences and embeddings list
                        for p_id, sentences in doc_sentences.items():
                            # preprocess each sentence
                            sent = []
                            for s in sentences:
                                s = self.preprocess_for_biosentvec(s)
                                sent.append(s)
                            processed_all_sentences.extend(sent)

                            all_sentences.extend(sentences)
                            sentence_embeddings[p_id] = [0] * len(sentences)
                        # computing all embeddings at once
                        print('{}: Computing sentence embeddings using BioSentVec for #{}'.format(time.time() - ttt, i + 1))
                        all_embeddings = list(self.sentence_model.embed_sentences(processed_all_sentences))
                    elif self._should_compute_any(gis_config, ['sen_model_mistral']):
                        # initializing sentences and embeddings list
                        for p_id, sentences in doc_sentences.items():
                            all_sentences.extend(sentences)
                            sentence_embeddings[p_id] = [0] * len(sentences)
                        # computing all embeddings at once
                        print('{}: Computing sentence embeddings for #{}'.format(time.time() - ttt, i + 1))   
                        all_embeddings = list(self.LLM_embed(self.LLM_sentence_tokenizer, self.LLM_sentence_model, all_sentences))       
                    else:
                        # initializing sentences and embeddings list
                        for p_id, sentences in doc_sentences.items():
                            all_sentences.extend(sentences)
                            sentence_embeddings[p_id] = [0] * len(sentences)
                        # computing all embeddings at once
                        print('{}: Computing sentence embeddings for #{}'.format(time.time() - ttt, i + 1))
                        all_embeddings = list(self.sentence_model.encode(all_sentences))

                    s_index = 0
                    for p_id, sentences in doc_sentences.items():
                        for idx, sentence in enumerate(sentences):
                            if sentence == all_sentences[s_index]:
                                sentence_embeddings[p_id][idx] = all_embeddings[s_index]
                                s_index += 1

                    print('{}: Computing PCREF for #{}'.format(time.time() - ttt, i + 1))
                    PCREF1, PCREFa, PCREF1p, PCREFap = self._compute_PCREF(sentence_embeddings)

                SEMCHUNK_mxbai,SEMCHUNK_cse, SEMCHUNK_biobert, SEMCHUNK_mistral = (0, 0, 0, 0)
                if self._should_compute_any(gis_config, ['SEMCHUNK_mxbai']):
                    print('{}: Computing SEMCHUNK_mxbai for #{}'.format(time.time() - ttt, i + 1))
                    SEMCHUNK_mxbai = self._compute_SEMCHUNK(doc_text, self.semchunk_splitter)
                if self._should_compute_any(gis_config, ['SEMCHUNK_cse']):
                    print('{}: Computing SEMCHUNK_cse for #{}'.format(time.time() - ttt, i + 1))
                    SEMCHUNK_cse = self._compute_SEMCHUNK(doc_text, self.semchunk_splitter)
                if self._should_compute_any(gis_config, ['SEMCHUNK_biobert']):
                    print('{}: Computing SEMCHUNK_biobert for #{}'.format(time.time() - ttt, i + 1))
                    SEMCHUNK_biobert = self._compute_SEMCHUNK(doc_text, self.semchunk_splitter)
                if self._should_compute_any(gis_config, ['SEMCHUNK_mistral']):
                    print('{}: Computing SEMCHUNK_mistral for #{}'.format(time.time() - ttt, i + 1))
                    SEMCHUNK_mistral = self._compute_SEMCHUNK(doc_text, self.semchunk_splitter)

                SMCAUSe_1, SMCAUSe_a, SMCAUSe_1p, SMCAUSe_ap = (0, 0, 0, 0)
                if self._should_compute_any(gis_config, ['SMCAUSe_1', 'SMCAUSe_a', 'SMCAUSe_1p', 'SMCAUSe_ap']):
                    print('{}: Computing SMCAUSe for #{}'.format(time.time() - ttt, i + 1))
                    SMCAUSe_1, SMCAUSe_a, SMCAUSe_1p, SMCAUSe_ap = self._compute_SMCAUSe(df_doc,
                                                                                        token_embeddings,
                                                                                        token_ids_by_sentence)

                SMCAUSf_1, SMCAUSf_a, SMCAUSf_1p, SMCAUSf_ap = (0, 0, 0, 0)
                if self._should_compute_any(gis_config, ['SMCAUSf_1', 'SMCAUSf_a', 'SMCAUSf_1p', 'SMCAUSf_ap']):
                    print('{}: Computing SMCAUSf for #{}'.format(time.time() - ttt, i + 1))
                    SMCAUSf_1, SMCAUSf_a, SMCAUSf_1p, SMCAUSf_ap = self._compute_SMCAUSe(df_doc,
                                                                                        token_embeddings,
                                                                                        token_ids_by_sentence)
                    
                SMCAUSb_1, SMCAUSb_a, SMCAUSb_1p, SMCAUSb_ap = (0, 0, 0, 0)
                if self._should_compute_any(gis_config, ['SMCAUSb_1', 'SMCAUSb_a', 'SMCAUSb_1p', 'SMCAUSb_ap']):
                    print('{}: Computing SMCAUSb for #{}'.format(time.time() - ttt, i + 1))
                    SMCAUSb_1, SMCAUSb_a, SMCAUSb_1p, SMCAUSb_ap = self._compute_SMCAUSe(df_doc,
                                                                                        token_embeddings,
                                                                                        token_ids_by_sentence)

                SMCAUSwn = {'SMCAUSwn_1p_path': 0, 'SMCAUSwn_1p_lch': 0,
                            'SMCAUSwn_1p_wup': 0, 'SMCAUSwn_1p_binary': 0,
                            'SMCAUSwn_ap_path': 0, 'SMCAUSwn_ap_lch': 0,
                            'SMCAUSwn_ap_wup': 0, 'SMCAUSwn_ap_binary': 0,
                            'SMCAUSwn_1_path': 0, 'SMCAUSwn_1_lch': 0,
                            'SMCAUSwn_1_wup': 0, 'SMCAUSwn_1_binary': 0,
                            'SMCAUSwn_a_path': 0, 'SMCAUSwn_a_lch': 0,
                            'SMCAUSwn_a_wup': 0, 'SMCAUSwn_a_binary': 0}
                if self._should_compute_any(gis_config, ['SMCAUSwn_1p_path', 'SMCAUSwn_1p_lch',
                                                        'SMCAUSwn_1p_wup', 'SMCAUSwn_1p_binary',
                                                        'SMCAUSwn_ap_path', 'SMCAUSwn_ap_lch',
                                                        'SMCAUSwn_ap_wup', 'SMCAUSwn_ap_binary',
                                                        'SMCAUSwn_1_path', 'SMCAUSwn_1_lch',
                                                        'SMCAUSwn_1_wup', 'SMCAUSwn_1_binary',
                                                        'SMCAUSwn_a_path', 'SMCAUSwn_a_lch',
                                                        'SMCAUSwn_a_wup', 'SMCAUSwn_a_binary']):
                    print('{}: Computing SMCAUSwn for #{}'.format(time.time() - ttt, i + 1))
                    SMCAUSwn = self._compute_SMCAUSwn(df_doc, token_ids_by_sentence)

                PCDC = 0
                if self._should_compute_any(gis_config, ['PCDC']):
                    print('{}: Computing PCDC for #{}'.format(time.time() - ttt, i + 1))
                    PCDC = self._compute_PCDC(doc_sentences)

                WRDCNCc_megahr, WRDIMGc_megahr = (0, 0)
                if self._should_compute_any(gis_config, ['PCCNC_megahr', 'WRDIMGc_megahr']):
                    print('{}: Computing *_megahr for #{}'.format(time.time() - ttt, i + 1))
                    WRDCNCc_megahr, WRDIMGc_megahr = self._compute_WRDCNCc_WRDIMGc_megahr(df_doc)

                WRDCNCc_mrc, WRDIMGc_mrc = (0, 0)
                if self._should_compute_any(gis_config, ['PCCNC_mrc', 'WRDIMGc_mrc']):
                    print('{}: Computing *_mrc for #{}'.format(time.time() - ttt, i + 1))
                    WRDCNCc_mrc, WRDIMGc_mrc = self._compute_WRDCNCc_WRDIMGc_mrc(df_doc)

                WRDHYPnv, WRDHYPnv_fixed, WRDHYPnv_fixed_mean, WRDHYPnv_fixed_min, WRDHYPnv_rootnorm = (0, 0, 0, 0, 0)
                if self._should_compute_any(gis_config, ['WRDHYPnv']):
                    print('{}: Computing WRDHYPnv for #{}'.format(time.time() - ttt, i + 1))
                    WRDHYPnv = self._compute_WRDHYPnv(df_doc, mode = 0)
                if self._should_compute_any(gis_config, ['WRDHYPnv_fixed']):
                    print('{}: Computing WRDHYPnv_fixed for #{}'.format(time.time() - ttt, i + 1))
                    WRDHYPnv_fixed = self._compute_WRDHYPnv(df_doc, mode = 1)
                if self._should_compute_any(gis_config, ['WRDHYPnv_fixed_mean']):
                    print('{}: Computing WRDHYPnv_fixed_mean for #{}'.format(time.time() - ttt, i + 1))
                    WRDHYPnv_fixed_mean = self._compute_WRDHYPnv(df_doc, mode = 2)
                if self._should_compute_any(gis_config, ['WRDHYPnv_fixed_min']):
                    print('{}: Computing WRDHYPnv_fixed_min for #{}'.format(time.time() - ttt, i + 1))
                    WRDHYPnv_fixed_min = self._compute_WRDHYPnv(df_doc, mode = 3)
                if self._should_compute_any(gis_config, ['WRDHYPnv_rootnorm']):
                    print('{}: Computing WRDHYPnv for #{}'.format(time.time() - ttt, i + 1))
                    WRDHYPnv_rootnorm = self._compute_WRDHYPnv(df_doc, mode = 4)

                WRDICnv = 0
                if self._should_compute_any(gis_config, ['WRDICnv']):
                    print('{}: Computing WRDICnv for #{}'.format(time.time() - ttt, i + 1))
                    WRDICnv = self._compute_WRDICnv(df_doc)

                print('#{} done'.format(i + 1))
                df_docs = df_docs.append(
                    {"d_id": txt_file, "text": doc_text, "DESPC": n_paragraphs, "DESSC": n_sentences,
                    "MSL": len(df_doc) / n_sentences,
                    "CoREF": CoREF, "PCREF_1": PCREF1, "PCREF_a": PCREFa, "PCREF_1p": PCREF1p,
                    "PCREF_ap": PCREFap,
                    "sen_model_mxbai": sen_model_mxbai,
                    "sen_model_cse": sen_model_cse,
                    "sen_model_biobert": sen_model_biobert,
                    "sen_model_mistral": sen_model_mistral,
                    "sen_model_biosentvec": sen_model_biosentvec,
                    "SEMCHUNK_mxbai": SEMCHUNK_mxbai,
                    "SEMCHUNK_cse": SEMCHUNK_cse, 
                    "SEMCHUNK_biobert": SEMCHUNK_biobert,
                    "SEMCHUNK_mistral": SEMCHUNK_mistral,
                    "PCDC": PCDC,
                    "SMCAUSe_1": SMCAUSe_1, "SMCAUSe_a": SMCAUSe_a, "SMCAUSe_1p": SMCAUSe_1p,
                    "SMCAUSe_ap": SMCAUSe_ap,
                    "SMCAUSf_1": SMCAUSf_1, "SMCAUSf_a": SMCAUSf_a, "SMCAUSf_1p": SMCAUSf_1p,
                    "SMCAUSf_ap": SMCAUSf_ap,
                    "SMCAUSb_1": SMCAUSb_1, "SMCAUSb_a": SMCAUSb_a, "SMCAUSb_1p": SMCAUSb_1p,
                    "SMCAUSb_ap": SMCAUSb_ap,
                    'SMCAUSwn_1p_path': SMCAUSwn['SMCAUSwn_1p_path'],
                    'SMCAUSwn_1p_lch': SMCAUSwn['SMCAUSwn_1p_lch'],
                    'SMCAUSwn_1p_wup': SMCAUSwn['SMCAUSwn_1p_wup'],
                    'SMCAUSwn_1p_binary': SMCAUSwn['SMCAUSwn_1p_binary'],
                    'SMCAUSwn_ap_path': SMCAUSwn['SMCAUSwn_ap_path'],
                    'SMCAUSwn_ap_lch': SMCAUSwn['SMCAUSwn_ap_lch'],
                    'SMCAUSwn_ap_wup': SMCAUSwn['SMCAUSwn_ap_wup'],
                    'SMCAUSwn_ap_binary': SMCAUSwn['SMCAUSwn_ap_binary'],
                    'SMCAUSwn_1_path': SMCAUSwn['SMCAUSwn_1_path'],
                    'SMCAUSwn_1_lch': SMCAUSwn['SMCAUSwn_1_lch'],
                    'SMCAUSwn_1_wup': SMCAUSwn['SMCAUSwn_1_wup'],
                    'SMCAUSwn_1_binary': SMCAUSwn['SMCAUSwn_1_binary'],
                    'SMCAUSwn_a_path': SMCAUSwn['SMCAUSwn_a_path'],
                    'SMCAUSwn_a_lch': SMCAUSwn['SMCAUSwn_a_lch'],
                    'SMCAUSwn_a_wup': SMCAUSwn['SMCAUSwn_a_wup'],
                    'SMCAUSwn_a_binary': SMCAUSwn['SMCAUSwn_a_binary'],
                    "PCCNC_megahr": WRDCNCc_megahr, "WRDIMGc_megahr": WRDIMGc_megahr,
                    "PCCNC_mrc": WRDCNCc_mrc, "WRDIMGc_mrc": WRDIMGc_mrc,
                    "WRDHYPnv": WRDHYPnv, 
                    "WRDHYPnv_fixed": WRDHYPnv_fixed, 
                    "WRDHYPnv_fixed_mean": WRDHYPnv_fixed_mean,
                    "WRDHYPnv_fixed_min": WRDHYPnv_fixed_min,
                    'WRDHYPnv_rootnorm': WRDHYPnv_rootnorm,
                    "WRDICnv": WRDICnv},
                    ignore_index=True)

        print('\n>>> computing indices for documents is done.')
        print('\n================= Report ==================')
        print('| # successfully completed documents | {}'.format(len(df_docs)))
        print('| # failed documents                 | {}'.format(len(errors_log)))
        print('===========================================\n')
        # writing error logs into a file
        if len(errors_log) > 0:
            with open(r'errors_log.txt', 'w') as fp:
                for error_log in errors_log:
                    fp.write("{}\n".format(error_log))
                print('>>> error logs written in the following file (if there was any error): errors_log.txt')

        return df_docs

    @staticmethod
    def _should_compute_any(gis_config, indices_list):
        for index in indices_list:
            if index in gis_config and gis_config[index] != 0:
                return True

        return False
    
    @staticmethod
    def preprocess_for_biosentvec(text):
        stop_words = set(stopwords.words('english'))
        text = text.replace('/', ' / ')
        text = text.replace('.-', ' .- ')
        text = text.replace('.', ' . ')
        text = text.replace('\'', ' \' ')
        text = text.lower()
        tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in stop_words]
        return ' '.join(tokens)
    


    @staticmethod
    def _filter_tokens_by_pos(df_doc, token_ids_by_sentence, pos_tags: list):
        tokens = dict()
        # keys: paragraph ids
        # values: one list of sentence embeddings for each paragraph id
        for p_s_id, token_ids in token_ids_by_sentence.items():
            p_id = p_s_id.split('_')[0]
            if p_id not in tokens:
                tokens[p_id] = list()
            current_tokens = list()
            for u_id in token_ids:
                row = df_doc.loc[df_doc['u_id'] == u_id]
                if row.iloc[0]['token_pos'] in pos_tags:
                    current_tokens.append({u_id: row.iloc['token_text']})
            if len(current_tokens) > 0:
                tokens[p_id].append(current_tokens)
        return tokens

    @staticmethod
    def _get_sentences_count(df_doc):
        """
        get the count of all sentences in a document
        :param df_doc:
        :return:
        """
        sents_count = 0
        paragraph_ids = df_doc['p_id'].unique()
        for p_id in paragraph_ids:
            paragraph_df = df_doc.loc[df_doc['p_id'] == p_id]
            paragraph_sents_count = len(paragraph_df['s_id'].unique())
            sents_count += paragraph_sents_count
        return sents_count

    def _get_doc_sentences(self, df_doc):
        """
        get list of sentences in a document
        :param df_doc:
        :return:
        """
        sentences = dict()
        current_sentence = str()
        df_doc.reset_index()
        p_ids = df_doc['p_id'].unique()
        for p_id in p_ids:
            sentences[p_id] = list()
            df_paragraph = df_doc.loc[df_doc['p_id'] == p_id]
            current_s_id = 0
            for idx, row in df_paragraph.iterrows():
                if row['s_id'] == current_s_id:
                    current_sentence += row['token_text'] + ' '
                else:
                    # end of current sentence, save it first
                    sentences[p_id].append(current_sentence.strip())
                    # reset variables for the next sentence
                    current_sentence = row['token_text'] + ' '
                    current_s_id += 1
            # saving the last sentence
            sentences[p_id].append(current_sentence.strip())
        len_sentences = sum([len(sentences[pid]) for pid in sentences.keys()])
        assert len_sentences == self._get_sentences_count(df_doc)
        return sentences, len(p_ids), len_sentences

    def _get_doc_token_ids_by_sentence(self, df_doc):
        """
        get list of token ids of sentences in a document
        :param df_doc:
        :return:
        """
        sentences_tokens = dict()
        df_doc.reset_index()
        p_ids = df_doc['p_id'].unique()
        for p_id in p_ids:
            current_sentence = list()
            df_paragraph = df_doc.loc[df_doc['p_id'] == p_id]
            current_s_id = 0
            for idx, row in df_paragraph.iterrows():
                if row['s_id'] == current_s_id:
                    current_sentence.append(row['u_id'])
                else:
                    # end of current sentence, save it first
                    sentences_tokens['{}_{}'.format(p_id, current_s_id)] = current_sentence
                    # reset variables for the next sentence
                    current_sentence = [row['u_id']]
                    current_s_id += 1
            # saving the last sentence
            sentences_tokens['{}_{}'.format(p_id, current_s_id)] = current_sentence
        tokens_count = sum([len(v) for k, v in sentences_tokens.items()])
        assert tokens_count == len(df_doc)
        return sentences_tokens

    @staticmethod
    def _find_causal_verbs(df_doc):
        causal_verbs = list()
        verbs = list(df_doc.loc[df_doc['token_pos'] == 'VERB'].token_lemma)
        for verb in verbs:
            verb_synsets = list(set(wn.synsets(verb, wn.VERB)))
            for verb_synset in verb_synsets:
                # checking if this verb can cause or entail anything
                if len(verb_synset.causes()) > 0 or len(verb_synset.entailments()) > 0:
                    causal_verbs.append(verb)
                    break  # we break here since for now we only want to know whether this verb can be causal at all.
        return causal_verbs, len(causal_verbs) / len(verbs)

    def _compute_SEMCHUNK(self, doc_text, splitter):
        """
        New submetric under PCREF: nunber of semantic chunks in the text, calculated by llama_index.semantic_chunking
        :return:
        """
        chunks = splitter.split_text(doc_text)
        return len(chunks)
        

    def _compute_PCDC(self, sentences):
        """
        finding the number of causal connectives in sentences in a document
        :return:
        """
        n_causal_connectives = 0
        matched_patterns = list()
        for p_id, p_sentences in sentences.items():
            for sentence in p_sentences:
                for pattern in self.causal_patterns:
                    if bool(pattern.match(sentence.lower())):
                        n_causal_connectives += 1
                        matched_patterns.append(pattern)
        sentences_count = sum([len(sentences[p_id]) for p_id in sentences.keys()])
        return n_causal_connectives / sentences_count

    def _compute_SMCAUSwn_v1(self, df_doc, similarity_measure='path'):
        """
        computing the WordNet Verb Overlap in a document
        :param similarity_measure: the type of similarity to use, one of the following: ['path', 'lch', 'wup]
        :return:
        """
        # getting all VERBs in a document
        verbs = list(df_doc.loc[df_doc['token_pos'] == 'VERB'].token_lemma)
        verb_synsets = dict()

        # getting all synsets (synonym sets) to which a verb belongs
        for verb in verbs:
            # check if synset is already in dictionary to avoid calling WordNet
            if verb in self.all_synsets:
                verb_synsets[verb] = self.all_synsets[verb]
            else:
                synsets = set(wn.synsets(verb, wn.VERB))
                self.all_synsets[verb] = synsets
                verb_synsets[verb] = synsets

        verb_pairs = itertools.combinations(verbs, r=2)

        similarity_scores = list()

        # computing the similarity of verb pairs by computing the average similarity between their synonym sets
        # each verb can have one or multiple synonym sets
        for verb_pair in verb_pairs:
            synset_pairs = itertools.product(verb_synsets[verb_pair[0]], verb_synsets[verb_pair[1]])
            for synset_pair in synset_pairs:
                if similarity_measure == 'path':
                    similarity_score = wn.path_similarity(synset_pair[0], synset_pair[1])
                elif similarity_measure == 'lch':
                    similarity_score = wn.lch_similarity(synset_pair[0], synset_pair[1])
                elif similarity_measure == 'wup':
                    similarity_score = wn.wup_similarity(synset_pair[0], synset_pair[1])

                # check if similarity_score is not None and is a number
                if isinstance(similarity_score, numbers.Number):
                    similarity_scores.append(similarity_score)

        return statistics.mean(similarity_scores)

    def _compute_WRDHYPnv(self, df_doc, mode = 0):
        """
        computing the specificity of a word within the WordNet hierarchy
        
        mode =
            0: original solution
            1: old fixed solution by maximun hypernym path
            2: fixed solution by mean hypernym path
            3: fixed solution by min hypernym path
            4: use longest hyoernym path & verb normalize on grouping by hypernym root
        """
        # getting all VERBs and NOUNs in document
        verbs_nouns = df_doc.loc[(df_doc['token_pos'] == 'VERB') | (df_doc['token_pos'] == 'NOUN')][
            ['token_text', 'token_lemma', 'token_pos']]

        if mode == 4:
            n_scores = []
            hp_lens_by_root = {}
            for _, row in verbs_nouns.iterrows():
                if row['token_pos'] == 'NOUN':
                    synsets = list(set(wn.synsets(row['token_text'], wn.NOUN)))
                    hypernym_path_length = 0
                    for synset in synsets:
                        hypernym_path_length += min(map(lambda path: len(path), synset.hypernym_paths()))
                        if len(synsets) > 0:
                            hypernym_score = hypernym_path_length / len(synsets)
                            n_scores.append(hypernym_score)

                elif row['token_pos'] == 'VERB':
                    synsets = list(set(wn.synsets(row['token_text'], wn.VERB)))
                    hypernym_path_length = 0
                    for synset in synsets:
                        hypernym_path_length += min(map(lambda path: len(path), synset.hypernym_paths()))
                        max_len_path = synset.hypernym_paths()[np.argmin(map(lambda path: len(path), synset.hypernym_paths()))]
                        root = max_len_path[0]
                        if root in hp_lens_by_root.keys():
                            hp_lens_by_root[root].append(hypernym_path_length)
                        else:
                            hp_lens_by_root[root] = [hypernym_path_length]
                            
            v_scores = hp_lens_by_root.values()

            n_scores_norm = [float(i)/sum(n_scores) for i in n_scores]
            v_scores_norm = []

            for v_score in v_scores:
                v_scores_norm.append([float(i)/sum(v_score) for i in v_score])

            n_scores_norm.extend([x for xs in v_scores_norm for x in xs])
            return statistics.mean(n_scores_norm)

        else:
            scores = list()
            for _, row in verbs_nouns.iterrows():
                try:
                    if row['token_pos'] == 'VERB':
                        synsets = list(set(wn.synsets(row['token_text'], wn.VERB)))
                    elif row['token_pos'] == 'NOUN':
                        synsets = list(set(wn.synsets(row['token_text'], wn.NOUN)))

                    hypernym_path_length = 0

                    for synset in synsets:
                        if mode == 0:
                            hypernym_path_length += len(synset.hypernym_paths())
                        elif mode == 1:
                            hypernym_path_length += max(map(lambda path: len(path), synset.hypernym_paths()))
                        elif mode == 2:
                            hypernym_path_length += statistics.mean(map(lambda path: len(path), synset.hypernym_paths()))
                        elif mode == 3:
                            hypernym_path_length += min(map(lambda path: len(path), synset.hypernym_paths()))
                    # computing the average length of hypernym path
                    hypernym_score = hypernym_path_length / len(synsets)
                    scores.append(hypernym_score)
                except:
                    pass
            return statistics.mean(scores)


    def _compute_WRDICnv(self, df_doc):
        """
        computing the information content
        :return:
        """
        # getting all VERBs and NOUNs in document
        verbs_nouns = df_doc.loc[(df_doc['token_pos'] == 'VERB') | (df_doc['token_pos'] == 'NOUN')][
            ['token_text', 'token_lemma', 'token_pos']]

        scores = list()
        for index, row in verbs_nouns.iterrows():
            try:
                pos = wn.VERB if row['token_pos'] == 'VERB' else wn.NOUN
                synsets = list(set(wn.synsets(row['token_text'], pos)))

                ic_all_synsets = 0
                for synset in synsets:
                    # formula from:
                    # https://raw.githubusercontent.com/nltk/wordnet/ed82598ce1bb09fedb4b9f334e591356eb59359f/wn/info.py
                    counts = self.ic[pos][synset.offset()]
                    counts = max(1.0, counts)

                    ic_all_synsets += -math.log(counts / self.max_ic_counts[pos])

                ic = ic_all_synsets / len(synsets)
                scores.append(ic)
            except:
                pass
        return statistics.mean(scores)

    def _compute_WRDCNCc_WRDIMGc_mrc(self, df_doc):
        """
        computing the document concreteness and imageability
        :return:
        """
        concreteness_scores = list()
        imageability_scores = list()
        for index, row in df_doc.iterrows():
            records = find_mrc_word(row['token_text'], row['token_pos'])
            # there might be more than one record for the very word with its POS tag
            if len(records) > 0:
                for record in records:
                    concreteness_scores.append(record.conc)
                    imageability_scores.append(record.imag)
        return statistics.mean(concreteness_scores), statistics.mean(imageability_scores)

    def _compute_WRDCNCc_WRDIMGc_megahr(self, df_doc):
        """
        computing the document concreteness and imageability
        :return:
        """
        concreteness_scores = list()
        imageability_scores = list()

        # filtering out tokens we don't need
        # pos_filter = ['NUM', 'PUNCT', 'SYM']
        pos_filter = []
        df_doc = df_doc.loc[~df_doc['token_pos'].isin(pos_filter)]

        for index, row in df_doc.iterrows():
            token_text = str(row['token_text']).lower()
            if token_text in self.megahr_dict:
                concreteness_scores.append(self.megahr_dict[token_text][0])
                imageability_scores.append(self.megahr_dict[token_text][1])

        if len(concreteness_scores) > 0 and len(imageability_scores) > 0:
            return statistics.mean(concreteness_scores), statistics.mean(imageability_scores)
        else:
            return None, None

    def _compute_SMCAUSe(self, df_doc, token_embeddings, token_ids_by_sentence, pos_tags=['VERB']):
        """
        computing the similarity among tokens with certain POS tag in a document
        *e* at the end stands for Embedding to show this method is a replacement for Latent Semantic Analysis (LSA) here
        :param df_doc: data frame of a document
        :param pos_tags: list of part-of-speech tags for which we want to compute the cosine similarity
        :return:
        """
        # we use this dictionary to avoid computing cosine for the very pair multiple times
        # by simply caching the cosine value of token pairs
        tokens_similarity = dict()

        def local_cosine(e):
            """
            e is a list of embeddings
            :param e:
            :return:
            """
            if len(e) <= 1:
                return 0
            else:
                j = 0
                scores = list()
                while j + 1 < len(e):
                    pair_id = '{}@{}'.format(list(e[j].keys())[0], list(e[j + 1].keys())[0])
                    if pair_id not in tokens_similarity:
                        score = util.cos_sim(list(e[j].values())[0], list(e[j + 1].values())[0]).item()
                        tokens_similarity[pair_id] = score
                    else:
                        score = tokens_similarity[pair_id]
                    scores.append(score)
                    j += 1
                return statistics.mean(scores)

        def global_cosine(pairs):
            scores = list()
            for pair in pairs:
                pair_id = '{}@{}'.format(list(pair[0].keys())[0], list(pair[1].keys())[0])
                if pair_id not in tokens_similarity:
                    score = util.cos_sim(list(pair[0].values())[0], list(pair[1].values())[0]).item()
                    tokens_similarity[pair_id] = score
                else:
                    score = tokens_similarity[pair_id]
                scores.append(score)
            return statistics.mean(scores) if len(scores) > 0 else 0

        embeddings = dict()
        # keys: paragraph ids
        # values: one list of sentence embeddings for each paragraph id
        for p_s_id, token_ids in token_ids_by_sentence.items():
            p_id = p_s_id.split('_')[0]
            if p_id not in embeddings:
                embeddings[p_id] = list()
            current_embeddings = list()
            for u_id in token_ids:
                row = df_doc.loc[df_doc['u_id'] == u_id]
                if row.iloc[0]['token_pos'] in pos_tags:
                    current_embeddings.append({u_id: token_embeddings[u_id]})
            if len(current_embeddings) > 0:
                embeddings[p_id].append(current_embeddings)

        scores_1p = list()
        scores_ap = list()

        token_embeddings_flat = list()

        for p_id, s_embeddings in embeddings.items():
            # *** consecutive cosine ***
            if len(s_embeddings) <= 1:
                scores_1p.append(0)
            else:
                i = 0
                while i + 1 < len(s_embeddings):
                    all_pairs = list(itertools.product(s_embeddings[i], s_embeddings[i + 1]))
                    scores_1p.append(global_cosine(all_pairs))
                    i += 1

            # *** global cosine ***
            t_embeddings = list()  # all token embeddings of all tokens in one paragraph
            for item in s_embeddings:
                t_embeddings.extend(item)
                token_embeddings_flat.extend(item)
            all_pairs = itertools.combinations(t_embeddings, r=2)
            scores_ap.append(global_cosine(all_pairs))

        SMCAUSe_1p = statistics.mean(scores_1p)
        SMCAUSe_ap = statistics.mean(scores_ap)

        # computing global and local indices ignoring the paragraphs
        all_pairs = itertools.combinations(token_embeddings_flat, r=2)
        SMCAUSe_a = global_cosine(all_pairs)
        SMCAUSe_1 = local_cosine(token_embeddings_flat)

        return SMCAUSe_1, SMCAUSe_a, SMCAUSe_1p, SMCAUSe_ap

    def _compute_SMCAUSwn(self, df_doc, token_ids_by_sentence, pos_tags=['VERB']):
        """
        computing WordNet Verb Overlap
        :param df_doc: data frame of a document
        :param token_ids_by_sentence:
        :param pos_tags: list of part-of-speech tags for which we want to compute the overlap
        :return:
        """

        # TODO: this method is a bottleneck in the pipeline. One reason is that for one verb we have too many synsets
        # TODO: we need to figure out how to find the most relevant synsets for a verb
        scores_functions = {'path': wn.path_similarity, 'lch': wn.lch_similarity, 'wup': wn.wup_similarity}

        def _build_result(scores):
            return {'path': statistics.mean(scores['path']) if len(scores['path']) > 0 else 0,
                    'lch': statistics.mean(scores['lch']) if len(scores['lch']) > 0 else 0,
                    'wup': statistics.mean(scores['wup']) if len(scores['wup']) > 0 else 0,
                    'binary': statistics.mean(scores['binary']) if len(scores['binary']) > 0 else 0}

        def synset_pair_similarity(pair):
            scores = {'path': list(), 'lch': list(), 'wup': list(), 'binary': list()}
            token_a = pair[0][0]
            token_a_synsets = pair[0][1]
            token_b = pair[1][0]
            token_b_synsets = pair[1][1]
            binary = 0
            if len(list(set(token_a_synsets).intersection(set(token_b_synsets)))) > 0:
                binary = 1
            verb_pair = '{}@{}'.format(token_a, token_b)
            if verb_pair in self.verb_similarity:
                return {'path': self.verb_similarity[verb_pair]['path'],
                        'lch': self.verb_similarity[verb_pair]['lch'],
                        'wup': self.verb_similarity[verb_pair]['wup'],
                        'binary': self.verb_similarity[verb_pair]['binary']}
            else:
                synset_pairs = itertools.product(token_a_synsets, token_b_synsets)
                for synset_pair in synset_pairs:
                    for score_name, score_function in scores_functions.items():
                        score = score_function(synset_pair[0], synset_pair[1])
                        scores[score_name].append(score)
                    scores['binary'] = [binary]
                # updating the dict to cache the result
                result = _build_result(scores)
                self.verb_similarity[verb_pair] = result
                return result

        def local_wn_cosine(synsets):
            """
            :param synsets:
            :return:
            """
            similarity_scores = {'path': list(), 'lch': list(), 'wup': list(), 'binary': list()}
            if len(synsets) <= 1:
                return {'path': 0, 'lch': 0, 'wup': 0, 'binary': 0}
            else:
                j = 0
                while j + 1 < len(synsets):
                    result = synset_pair_similarity((synsets[j], synsets[j + 1]))
                    for score_name in similarity_scores.keys():
                        similarity_scores[score_name].append(result[score_name])
                    j += 1
                return _build_result(similarity_scores)

        def global_wn_overlap(pairs):
            """
            computing the wordnet verb overlap among pairs
            :param pairs: list of pair items where each pair has two elements where each element is a list of synsets
            :return:
            """
            similarity_scores = {'path': list(), 'lch': list(), 'wup': list(), 'binary': list()}
            for pair in pairs:
                result = synset_pair_similarity(pair)
                for score_name in result.keys():
                    similarity_scores[score_name].append(result[score_name])

            return _build_result(similarity_scores)

        token_synsets = dict()
        for p_s_id, token_ids in token_ids_by_sentence.items():
            p_id = p_s_id.split('_')[0]
            if p_id not in token_synsets:
                token_synsets[p_id] = list()
            current_synsets = list()
            for u_id in token_ids:
                row = df_doc.loc[df_doc['u_id'] == u_id]
                if row.iloc[0]['token_pos'] in pos_tags:
                    token = row.iloc[0]['token_text']
                    synsets = set(wn.synsets(token, wn.VERB))
                    if len(synsets) > 0:
                        current_synsets.append([token, synsets])
            if len(current_synsets) > 0:
                token_synsets[p_id].append(current_synsets)

        scores_1p = list()
        scores_ap = list()

        synsets_flat = list()

        for p_id, synsets_by_sentences in token_synsets.items():
            # *** consecutive (local) cosine ***
            if len(synsets_by_sentences) <= 1:
                scores_1p.append({'path': 0, 'lch': 0, 'wup': 0, 'binary': 0})
            else:
                i = 0
                while i + 1 < len(synsets_by_sentences):
                    pairs = list(itertools.product(synsets_by_sentences[i], synsets_by_sentences[i + 1]))
                    scores_1p.append(global_wn_overlap(pairs))
                    i += 1

            # *** global cosine ***
            t_synsets = list()  # all synsets of all tokens in one paragraph
            for item in synsets_by_sentences:
                t_synsets.extend(item)
                synsets_flat.extend(item)
            all_pairs = itertools.combinations(t_synsets, r=2)
            scores_ap.append(global_wn_overlap(all_pairs))

        # computing global and local indices ignoring the paragraphs
        all_pairs = itertools.combinations(synsets_flat, r=2)
        SMCAUSwn_a = global_wn_overlap(all_pairs)
        SMCAUSwn_1 = local_wn_cosine(synsets_flat)

        return {'SMCAUSwn_1p_path': statistics.mean([item['path'] for item in scores_1p]),
                'SMCAUSwn_1p_lch': statistics.mean([item['lch'] for item in scores_1p]),
                'SMCAUSwn_1p_wup': statistics.mean([item['wup'] for item in scores_1p]),
                'SMCAUSwn_1p_binary': statistics.mean([item['binary'] for item in scores_1p]),
                'SMCAUSwn_ap_path': statistics.mean([item['path'] for item in scores_ap]),
                'SMCAUSwn_ap_lch': statistics.mean([item['lch'] for item in scores_ap]),
                'SMCAUSwn_ap_wup': statistics.mean([item['wup'] for item in scores_ap]),
                'SMCAUSwn_ap_binary': statistics.mean([item['binary'] for item in scores_ap]),
                'SMCAUSwn_1_path': SMCAUSwn_1['path'],
                'SMCAUSwn_1_lch': SMCAUSwn_1['lch'],
                'SMCAUSwn_1_wup': SMCAUSwn_1['wup'],
                'SMCAUSwn_1_binary': SMCAUSwn_1['binary'],
                'SMCAUSwn_a_path': SMCAUSwn_a['path'],
                'SMCAUSwn_a_lch': SMCAUSwn_a['lch'],
                'SMCAUSwn_a_wup': SMCAUSwn_a['wup'],
                'SMCAUSwn_a_binary': SMCAUSwn_a['binary']}

    def _compute_PCREF(self, sentence_embeddings):
        """
        Computing Text Easability PC Referential cohesion
        :param sentence_embeddings: embeddings of sentences in a document
        :return:
        """

        # local: only consecutive sentences either in a paragraph or entire text
        # global: all sentence pairs in a paragraph or entire text
        all_embeddings = list()

        # flattening the embedding list
        for p_id, embeddings in sentence_embeddings.items():
            for embedding in embeddings:
                all_embeddings.append(embedding)

        local_cosine = self._local_cosine(all_embeddings)
        global_cosine = self._global_cosine(all_embeddings)

        del all_embeddings

        local_scores = dict()
        global_scores = dict()
        for p_id, embeddings in sentence_embeddings.items():
            local_scores[p_id] = self._local_cosine(embeddings)
            global_scores[p_id] = self._global_cosine(embeddings)

        # *_p means computed at paragraph-level in contrast to the first case where we ignored paragraphs
        local_cosine_p = statistics.mean([local_scores[p_id] for p_id in local_scores.keys()])
        global_cosine_p = statistics.mean([global_scores[p_id] for p_id in global_scores.keys()])

        return local_cosine, global_cosine, local_cosine_p, global_cosine_p
    
    @staticmethod
    def LLM_embed(tokenizer, model, text): 

        def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
            
        max_length = 4096
        # Tokenize the input texts
        batch_dict = tokenizer(text, max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True)
        # append eos_token_id to every input_ids
        batch_dict['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
        batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt')

        with torch.no_grad():
            outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        return embeddings



class GIS:
    def __init__(self):
        self.wolfe_mean_sd = {'SMCAUSlsa': {'mean': 0.097, 'sd': 0.04},
                              'SMCAUSwn': {'mean': 0.553, 'sd': 0.096},
                              'WRDIMGc': {'mean': 410.346, 'sd': 24.994},
                              'WRDHYPnv': {'mean': 1.843, 'sd': 0.26}}
        self.gispy_columns = GisPyData().get_gispy_index_columns()
        self.cohmetrix_columns = ["SMCAUSlsa", "SMCAUSwn", "WRDIMGc", "WRDHYPnv"]

    def _z_score(self, df, index_name, wolfe=False):
        if index_name.startswith('sen_model_'):
            return 0
        if wolfe:
            return df[index_name].map(
                lambda x: (x - self.wolfe_mean_sd[index_name]['mean']) / self.wolfe_mean_sd[index_name]['sd'])
        else:
            return stats.zscore(df[index_name].astype(float), nan_policy='omit')

    def score(self, df, variables: dict, wolfe=False, gispy=False):
        """
        computing Gist Inference Score (GIS) based on the following paper:
        https://link.springer.com/article/10.3758/s13428-019-01284-4
        use this method when values of indices are computed using CohMetrix
        :param df: a dataframe that contains coh-metrix indices
        :param variables: a dictionary of information of variables we need to compute the GIS score
        :param wolfe: whether using wolfe's mean and standard deviation for computing z-score
        :param gispy: whether indices are computed by gispy or not (if not gispy, indices should be computed by CohMetrix)
        :return: the input dataframe with an extra column named "GIS" that stores gist inference score
        """

        # Referential Cohesion (PCREFz)
        # Deep Cohesion (PCDCz)
        # Verb Overlap LSA (SMCAUSlsa)
        # Verb Overlap WordNet (SMCAUSwn)
        # Word Concreteness (PCCNCz)
        # Imageability for Content Words (WRDIMGc)
        # Hypernymy for Nouns and Verbs (WRDHYPnv)

        # Z-Score(X) = (X-μ)/σ
        # X: a single raw data value
        # μ: population mean
        # σ: population standard deviation

        columns = self.gispy_columns if gispy else self.cohmetrix_columns
        # columns = GisPyData().get_variables_dict(gispy=True).keys() 
        
        for column in columns:
            df["z{}".format(column)] = self._z_score(df, index_name=column, wolfe=wolfe)

        df['zero'] = 0
        # computing the Gist Inference Score (GIS)
        for idx, row in df.iterrows():
            gis = 0
            for variable_name, variable in variables.items():
                gis += variable['weight'] * statistics.mean([row[index_name] for index_name in variable['vars']])
            df.loc[idx, "gis"] = gis

        return df
