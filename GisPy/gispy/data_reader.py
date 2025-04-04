import spacy
from spacy.language import Language
import itertools
import numpy as np
import pandas as pd
import fasttext
from huggingface_hub import hf_hub_download

import os
from os import listdir
from os.path import isfile, join


@Language.factory('tensor2attr')
class Tensor2Attr:
    """
    source code: https://applied-language-technology.mooc.fi/html/notebooks/part_iii/05_embeddings_continued.html
    """

    # We continue by defining the first method of the class,
    # __init__(), which is called when this class is used for
    # creating a Python object. Custom components in spaCy
    # require passing two variables to the __init__() method:
    # 'name' and 'nlp'. The variable 'self' refers to any
    # object created using this class!
    def __init__(self, name, nlp):
        # We do not really do anything with this class, so we
        # simply move on using 'pass' when the object is created.
        pass

    # The __call__() method is called whenever some other object
    # is passed to an object representing this class. Since we know
    # that the class is a part of the spaCy pipeline, we already know
    # that it will receive Doc objects from the preceding layers.
    # We use the variable 'doc' to refer to any object received.
    def __call__(self, doc):
        # When an object is received, the class will instantly pass
        # the object forward to the 'add_attributes' method. The
        # reference to self informs Python that the method belongs
        # to this class.
        self.add_attributes(doc)

        # After the 'add_attributes' method finishes, the __call__
        # method returns the object.
        return doc

    # Next, we define the 'add_attributes' method that will modify
    # the incoming Doc object by calling a series of methods.
    def add_attributes(self, doc):
        # spaCy Doc objects have an attribute named 'user_hooks',
        # which allows customising the default attributes of a
        # Doc object, such as 'vector'. We use the 'user_hooks'
        # attribute to replace the attribute 'vector' with the
        # Transformer output, which is retrieved using the
        # 'doc_tensor' method defined below.
        doc.user_hooks['vector'] = self.doc_tensor

        # We then perform the same for both Spans and Tokens that
        # are contained within the Doc object.
        doc.user_span_hooks['vector'] = self.span_tensor
        doc.user_token_hooks['vector'] = self.token_tensor

        # We also replace the 'similarity' method, because the
        # default 'similarity' method looks at the default 'vector'
        # attribute, which is empty! We must first replace the
        # vectors using the 'user_hooks' attribute.
        doc.user_hooks['similarity'] = self.get_similarity
        doc.user_span_hooks['similarity'] = self.get_similarity
        doc.user_token_hooks['similarity'] = self.get_similarity

    # Define a method that takes a Doc object as input and returns
    # Transformer output for the entire Doc.
    def doc_tensor(self, doc):
        # Return Transformer output for the entire Doc. As noted
        # above, this is the last item under the attribute 'tensor'.
        # Average the output along axis 0 to handle batched outputs.
        return doc._.trf_data.tensors[-1].mean(axis=0)

    # Define a method that takes a Span as input and returns the Transformer
    # output.
    def span_tensor(self, span):
        # Get alignment information for Span. This is achieved by using
        # the 'doc' attribute of Span that refers to the Doc that contains
        # this Span. We then use the 'start' and 'end' attributes of a Span
        # to retrieve the alignment information. Finally, we flatten the
        # resulting array to use it for indexing.
        tensor_ix = span.doc._.trf_data.align[span.start: span.end].data.flatten()

        # Fetch Transformer output shape from the final dimension of the output.
        # We do this here to maintain compatibility with different Transformers,
        # which may output tensors of different shape.
        out_dim = span.doc._.trf_data.tensors[0].shape[-1]

        # Get Token tensors under tensors[0]. Reshape batched outputs so that
        # each "row" in the matrix corresponds to a single token. This is needed
        # for matching alignment information under 'tensor_ix' to the Transformer
        # output.
        tensor = span.doc._.trf_data.tensors[0].reshape(-1, out_dim)[tensor_ix]

        # Average vectors along axis 0 ("columns"). This yields a 768-dimensional
        # vector for each spaCy Span.
        return tensor.mean(axis=0)

    # Define a function that takes a Token as input and returns the Transformer
    # output.
    def token_tensor(self, token):
        # Get alignment information for Token; flatten array for indexing.
        # Again, we use the 'doc' attribute of a Token to get the parent Doc,
        # which contains the Transformer output.
        tensor_ix = token.doc._.trf_data.align[token.i].data.flatten()

        # Fetch Transformer output shape from the final dimension of the output.
        # We do this here to maintain compatibility with different Transformers,
        # which may output tensors of different shape.
        out_dim = token.doc._.trf_data.tensors[0].shape[-1]

        # Get Token tensors under tensors[0]. Reshape batched outputs so that
        # each "row" in the matrix corresponds to a single token. This is needed
        # for matching alignment information under 'tensor_ix' to the Transformer
        # output.
        tensor = token.doc._.trf_data.tensors[0].reshape(-1, out_dim)[tensor_ix]

        # Average vectors along axis 0 (columns). This yields a 768-dimensional
        # vector for each spaCy Token.
        return tensor.mean(axis=0)

    # Define a function for calculating cosine similarity between vectors
    def get_similarity(self, doc1, doc2):
        # Calculate and return cosine similarity
        return np.dot(doc1.vector, doc2.vector) / (doc1.vector_norm * doc2.vector_norm)


nlp_trf = spacy.load('en_core_web_trf')
nlp_trf.add_pipe('tensor2attr')
nlp_trf.add_pipe('sentencizer')

fasttext_model_path = hf_hub_download(repo_id="facebook/fasttext-en-vectors", filename="model.bin")
fasttext_model = fasttext.load_model(fasttext_model_path)

# biowordvec_model_path = "BIOWORDPATH"
# biowordvec_model = fasttext.load_model(biowordvec_model_path)


def read_documents(docs_path):
    docs = []

    if os.path.isdir(docs_path):
        txt_files = [f for f in listdir(docs_path) if isfile(join(docs_path, f)) and '.txt' in f]
        print('total # of documents: {}'.format(len(txt_files)))
        print('computing indices for documents...')
        for i, txt_file in enumerate(txt_files):
            encodings = ['utf-8', 'iso-8859-1']
            doc_text = None

            # trying multiple encodings in case there's any error with one encoding
            for encoding in encodings:
                try:
                    with open('{}/{}'.format(docs_path, txt_file), encoding=encoding) as input_file:
                        doc_text = input_file.read()
                        break
                except UnicodeDecodeError:
                    continue

            # if the input file is successfully read, doc_text should not be empty.
            if doc_text is not None:
                docs.append((txt_file, doc_text))

        return docs

    else:
        raise Exception(
            'The document directory path you are using does not exist.\nCurrent path: {}'.format(docs_path))


class GisPyData:
    def __init__(self):
        pass

    @staticmethod
    def get_gispy_index_columns():
        return ["DESPC", "DESSC", "MSL",
                "CoREF", "PCREF_1", "PCREF_a", "PCREF_1p", "PCREF_ap", 
                "sen_model_mxbai", "sen_model_cse", "sen_model_biobert", "sen_model_mistral", "sen_model_biosentvec",
                "SEMCHUNK_mxbai", "SEMCHUNK_cse", "SEMCHUNK_biobert", "SEMCHUNK_mistral",
                "PCDC",
                "SMCAUSe_1", "SMCAUSe_a", "SMCAUSe_1p", "SMCAUSe_ap",
                "SMCAUSf_1", "SMCAUSf_a", "SMCAUSf_1p", "SMCAUSf_ap",
                "SMCAUSb_1", "SMCAUSb_a", "SMCAUSb_1p", "SMCAUSb_ap",
                "SMCAUSwn_1p_path", "SMCAUSwn_1p_lch", "SMCAUSwn_1p_wup",
                "SMCAUSwn_ap_path", "SMCAUSwn_ap_lch", "SMCAUSwn_ap_wup",
                "SMCAUSwn_1_path", "SMCAUSwn_1_lch", "SMCAUSwn_1_wup",
                "SMCAUSwn_a_path", "SMCAUSwn_a_lch", "SMCAUSwn_a_wup",
                "SMCAUSwn_1p_binary", "SMCAUSwn_ap_binary", "SMCAUSwn_1_binary", "SMCAUSwn_a_binary",
                "PCCNC_megahr", "WRDIMGc_megahr", "PCCNC_mrc", "WRDIMGc_mrc",
                "WRDHYPnv", "WRDHYPnv_fixed", "WRDHYPnv_fixed_mean", 
                "WRDHYPnv_fixed_min", "WRDHYPnv_rootnorm", "WRDICnv"]

    @staticmethod
    def convert_config_to_vars_dict(config_dict):
        vars_dict = dict()
        for index, (key, value) in enumerate(config_dict.items()):
            vars_dict["var{}".format(index)] = {'vars': ["z{}".format(key)], 'weight': value}
        return vars_dict

    @staticmethod
    def get_variables_dict(gispy=True, custom_vars=[]):
        """
        creating a dictionary of indices for GIS calculation
        :param gispy: binary to show either use GisPy or Coh-Metrix indices
        :param custom_vars:
        :return:
        """
        # weight: the weight associated with each index to enable weighted combination of indices
        # weight: 0 --> ignore the index
        # weight > 0 | weight < 0 --> include index and multiply it by the weight.
        # default gis formula = PCREFz + PCDCz + (zSMCAUSlsa - zSMCAUSwn) - PCCNCz - zWRDIMGc - zWRDHYPnv

        dicts = list()

        if gispy:
            vars_dict = {'var1': {'vars': ["PCREF_ap"], 'weight': 1},
                         'var2': {'vars': ["MSL"], 'weight': -1},
                         'var3': {'vars': ["PCDC"], 'weight': 1},
                         'var4': {'vars': ["SMCAUSe_1p"], 'weight': 1},
                         'var5': {'vars': ["SMCAUSwn_a_binary"], 'weight': -1},
                         'var6': {'vars': ["PCCNC_megahr"], 'weight': -1},
                         'var7': {'vars': ["WRDIMGc_megahr"], 'weight': -1},
                         'var8': {'vars': ["WRDICnv"], 'weight': -1}
                         }
            dicts.append(vars_dict)
            # if len(custom_vars) == 0:

            #     var1 = [#{'vars': ['zPCREF_ap'], 'weight': 0.6},
            #             {'vars': ['zPCREF_ap'], 'weight': 1},
            #             # {'vars': ['zero'], 'weight': 0}
            #             ]

            #     var3 = [#{'vars': ['zSMCAUSe_1p'], 'weight': 0.4},
            #             {'vars': ['zSMCAUSe_1p'], 'weight': 1},
            #             # {'vars': ['zero'], 'weight': 0}
            #             ]

            #     var4 = [#{'vars': ['zSMCAUSwn_a_binary'], 'weight': -0.6},
            #             {'vars': ['zSMCAUSwn_a_binary'], 'weight': -1},
            #             # {'vars': ['zero'], 'weight': 0}
            #             ]

            #     var5 = [#{'vars': ['zPCCNC_megahr'], 'weight': -0.4},
            #             {'vars': ['zPCCNC_megahr'], 'weight': -1},
            #             #{'vars': ['zero'], 'weight': 0}
            #             ]

            #     var6 = [#{'vars': ['zWRDIMGc_megahr'], 'weight': -0.4},
            #             {'vars': ['zWRDIMGc_megahr'], 'weight': -1},
            #             ]

            #     var7 = [{'vars': ['zWRDHYPnv'], 'weight': -1},
            #             {'vars': ['zWRDHYPnv_fixed'], 'weight': 1},
            #             {'vars': ['zWRDICnv'], 'weight': 1},
            #             {'vars': ['zero'], 'weight': 0}
            #             ]

            #     var7 = [
            #             {'vars': ['zero'], 'weight': 0},
            #             ]

            #     var8 = [#{'vars': ['zMSL'], 'weight': -0.4},
            #             {'vars': ['zMSL'], 'weight': -1},
            #             # {'vars': ['zero'], 'weight': 0}
            #             ]

            #     varPCDC = [#{'vars': ['zPCDC'], 'weight': 0.2},
            #                {'vars': ['zPCDC'], 'weight': 1}
            #                ]

            # else:
            #     var1 = [{'vars': [custom_vars[0]], 'weight': 1}]
            #     var3 = [{'vars': [custom_vars[1]], 'weight': 1}]
            #     var4 = [{'vars': [custom_vars[2]], 'weight': -1}]
            #     var5 = [{'vars': [custom_vars[3]], 'weight': -1}]
            #     var6 = [{'vars': [custom_vars[4]], 'weight': -1}]
            #     var7 = [{'vars': [custom_vars[5]], 'weight': -1}]
            #     var8 = [{'vars': [custom_vars[6]], 'weight': -1}]

            # products = list(itertools.product(var1, var3, var4, var5, var6, var7, var8, varPCDC))

            # for variables in products:
            #     vars_dict = {'var1': variables[0],
            #                  'var2': variables[7],
            #                  'var3': variables[1],
            #                  'var4': variables[2],
            #                  'var5': variables[3],
            #                  'var6': variables[4],
            #                  'var7': variables[5],
            #                  'var8': variables[6]}
            #     dicts.append(vars_dict)
        else:
            vars_dict = {'var1': {'vars': ['PCREFz'], 'weight': 1},
                         'var2': {'vars': ['PCDCz'], 'weight': 1},
                         'var3': {'vars': ['zSMCAUSlsa'], 'weight': 1},
                         'var4': {'vars': ['zSMCAUSwn'], 'weight': -1},
                         'var5': {'vars': ['PCCNCz'], 'weight': -1},
                         'var6': {'vars': ['zWRDIMGc'], 'weight': -1},
                         'var7': {'vars': ['zWRDHYPnv'], 'weight': -1}}
            dicts.append(vars_dict)

        return dicts

    @staticmethod
    def generate_variables_dict_id(variables_dict):
        vars_name_string = '#'.join([str(variables_dict[v]['weight']) + '#'.join(variables_dict[v]['vars']) for v in list(variables_dict.keys())])
        vars_name_list = vars_name_string.split('#')
        return vars_name_string, vars_name_list

    @staticmethod
    def convert_doc(doc_text, use_fasttext=False, use_biowordvec=False):
        """
        converting a document to tokens with meta-information (e.g. POS tags, vector embeddings)
        :param doc_text: text of a document
        :return:
        """
        if use_fasttext:
            fasttext_model_path = hf_hub_download(repo_id="facebook/fasttext-en-vectors", filename="model.bin")
            fasttext_model = fasttext.load_model(fasttext_model_path)
        elif use_biowordvec:
            biowordvec_model_path = "BioWordVec_PubMed_MIMICIII_d200.bin"
            biowordvec_model = fasttext.load_model(biowordvec_model_path)

        # u_id: unique identifier
        df_doc = pd.DataFrame(columns=["u_id", "p_id", "s_id", "token_id", "token_text", "token_lemma", "token_pos"])
        token_embeddings = dict()
        paragraphs = doc_text.split('\n')
        p_id = 0
        u_id = 0
        for paragraph in nlp_trf.pipe(paragraphs, disable=["parser"]):
            s_id = 0
            for sent in paragraph.sents:
                tokens = [t for t in sent]
                t_id = 0
                for token in tokens:
                    token_text = token.text.strip()
                    df_doc = df_doc.append({"u_id": u_id,
                                            "p_id": p_id,
                                            "s_id": s_id,
                                            "token_id": t_id,
                                            "token_text": token_text,
                                            "token_lemma": token.lemma_.strip(),
                                            "token_pos": token.pos_},
                                           ignore_index=True)
                    if use_fasttext:
                        token_embeddings[u_id] = fasttext_model[token_text.lower()]
                    elif use_biowordvec:
                        token_embeddings[u_id] = biowordvec_model[token_text.lower()]
                    else:
                        token_embeddings[u_id] = token.vector

                    u_id += 1
                    t_id += 1
                s_id += 1
            p_id += 1

        return df_doc, token_embeddings
