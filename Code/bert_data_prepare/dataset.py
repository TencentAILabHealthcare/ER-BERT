# -*- coding: utf-8 -*-

import os
import pandas as pd
from os import listdir
from tqdm import tqdm
from collections import OrderedDict
from os.path import join, exists
from utility import is_valid_aaseq

class BERTDataPrepare(object):
    def __init__(self, logger):
        self.data_dir = '../../RawData'
        self.save_dir = '../../ProcessedData'
        self.logger = logger
        self.columns = ['epitope', 'epitope_species', 'epitope_gene', 
                        'species', 'cdr3a', 'cdr3b', 'mhc_class', 
                        'source', 'ref_id']

    def _statistics(self, df):
        df_filter = df[['epitope', 'cdr3a', 'cdr3b']]

        epitope_df = df_filter.dropna(subset=['epitope'])
        cdr3a_df = df_filter.dropna(subset=['cdr3a'])
        cdr3b_df = df_filter.dropna(subset=['cdr3b'])
        self.logger.info('Unique epitope {}, cdr3 alpha {}, beta {}'.format(
            len(set(epitope_df['epitope'])), len(set(cdr3a_df['cdr3a'])), len(set(cdr3b_df['cdr3b']))))       

        def get_valid_df(col_list):
            df_col = df_filter.dropna(subset=col_list)[col_list]
            if len(df_col) == 0:
                return df_col
            if len(col_list) == 1:
                df_col = df_col[df_col[col_list[0]].map(is_valid_aaseq)]
            if len(col_list) == 2:
                df_col = df_col[(df_col[col_list[0]].map(is_valid_aaseq)) &
                                (df_col[col_list[1]].map(is_valid_aaseq))]
            if len(col_list) == 3:
                df_col = df_col[(df_col[col_list[0]].map(is_valid_aaseq)) &
                                (df_col[col_list[1]].map(is_valid_aaseq)) &
                                (df_col[col_list[2]].map(is_valid_aaseq))]
            return df_col

        valid_epitope = get_valid_df(col_list=['epitope'])
        valid_cdr3a = get_valid_df(col_list=['cdr3a'])
        valid_cdr3b = get_valid_df(col_list=['cdr3b'])
        self.logger.info('Valid epitope {}, cdr3 alpha {}, beta {}'.format(
                len(valid_epitope), len(valid_cdr3a), len(valid_cdr3b)))

        valid_epitope_cdr3a = get_valid_df(col_list=['epitope', 'cdr3a'])
        valid_epitope_cdr3b = get_valid_df(col_list=['epitope', 'cdr3b'])
        valid_epitope_cdr3a_cdr3b = get_valid_df(col_list=['epitope', 'cdr3a', 'cdr3b'])
        self.logger.info('Valid epitope-cdr3a {}, epitope-cdr3b {}, epitope-cdr3a-cdr3b {}\n'.format(
                len(valid_epitope_cdr3a), len(valid_epitope_cdr3b), len(valid_epitope_cdr3a_cdr3b)))

        return valid_epitope, valid_cdr3a, valid_cdr3b

    def _save_df(self, save_dir, df, epitope_df, cdr3a_df, cdr3b_df):
        df.to_csv(join(save_dir, 'full.csv'), index=False)
        if len(epitope_df) != 0:
            epitope_df = epitope_df.drop_duplicates()
            epitope_df.to_csv(join(save_dir, 'epitope.csv'), index=False)
        if len(cdr3a_df) != 0:
            cdr3a_df = cdr3a_df.drop_duplicates()
            cdr3a_df.to_csv(join(save_dir, 'alpha.csv'), index=False)
        if len(cdr3b_df) != 0:
            cdr3b_df = cdr3b_df.drop_duplicates()
            cdr3b_df.to_csv(join(save_dir, 'beta.csv'), index=False)

    def get_dataset(self, data_list):
        GENE_PLUS_DIR = '/aaa/louisyuzhao/project2/data_jiyinjia/files_from_scPlatform/zhaoyu_files/GenePlus_TCRdata/'
        MAB_DIR = '/aaa/louisyuzhao/project2/Microsoft_Adaptive_Biotechnologies_datasets/ImmuneCODE-Repertoires-002.2'
        total_data_dict = {'VDJdb': (join(self.data_dir, 'VDJdb/vdjdb_20210201.txt'), self.VDJdb_loader), 
                           'IEDB-Receptor': (join(self.data_dir, 'IEDB/iedb_receptor_full_v3.csv'), self.IEDB_Receptor_loader), 
                           'IEDB-Epitope': (join(self.data_dir, 'IEDB/epitope_full_v3.zip'), self.IEDB_Epitope_loader),
                           'GenePlus-Cancer': (join(GENE_PLUS_DIR, 'GenePlus_TCRdata.clinic_sequence.tsv'), self.GenePlus_Cancer_loader), 
                           'GenePlus-COVID': (join(GENE_PLUS_DIR, 'GenePlus_TCRdata.covid19_sequence.tsv'), self.GenePlus_COVID_loader),
                           'MAB-COVID': (join(MAB_DIR, ''), self.MAB_COVID_loader), 
                           'TCRdb': (join(self.data_dir, 'TCRdb'), self.TCRdb_loader),
                           'PIRD': (join(self.data_dir, 'PIRD/pird_tcr_ab.csv'), self.PIRD_loader),
                           'Glanville': (join(self.data_dir, 'Glanville/glanville_curated.csv'), self.Glanville_loader),
                           'Dash': (join(self.data_dir, 'Dash/human_mouse_pairseqs_v1_parsed_seqs_probs_mq20_clones.tsv'), self.Dash_loader),
                           'McPAS': (join(self.data_dir, 'McPAS/McPAS-TCR_20220728.csv'), self.McPAS_loader),
                           'NetTCR': (join(self.data_dir, 'NetTCR/train_ab_90_alphabeta.csv'), self.NetTCR_loader),
                           'huARdb': (join(self.data_dir, 'huARdb/20220817_HUARC_VDJ_hcT.csv'), self.huARdb_data_loader)}
        
        valid_epitope_list, valid_cdr3a_list, valid_cdr3b_list = [], [], []
        for data in data_list:
            self.logger.info(data)
            if exists(join(self.save_dir, data, 'full.csv')):
                df = pd.read_csv(join(self.save_dir, data, 'full.csv'))
                valid_epitope_df, valid_cdr3a_df, valid_cdr3b_df = self._statistics(df)
            else:
                save_dir = join(self.save_dir, data)
                os.makedirs(save_dir, exist_ok=True)
                df = total_data_dict[data][1](fn_source=total_data_dict[data][0])
                valid_epitope_df, valid_cdr3a_df, valid_cdr3b_df = self._statistics(df)
                self._save_df(save_dir, df=df, epitope_df=valid_epitope_df,
                              cdr3a_df=valid_cdr3a_df, cdr3b_df=valid_cdr3b_df)
            
            valid_epitope_list += list(valid_epitope_df['epitope'])
            valid_cdr3a_list += list(valid_cdr3a_df['cdr3a'])
            valid_cdr3b_list += list(valid_cdr3b_df['cdr3b'])

            del df
            del valid_epitope_df
            del valid_cdr3a_df
            del valid_cdr3b_df

        save_dir = join(self.save_dir, 'merged')
        if not exists(join(self.save_dir, 'merged', 'epitope.csv')):
            self.logger.info('Merged data not existed, create new one')
            os.makedirs(save_dir, exist_ok=True)
        else:
            valid_epitope_df = pd.read_csv(join(save_dir, 'epitope.csv'))
            self.logger.info(f'epitope.csv existed with length {len(valid_epitope_df)}')
            valid_epitope_list = list(set(valid_epitope_df['epitope']) | set(valid_epitope_list))
            self.logger.info(f'After adding new epitope data, current length {len(valid_epitope_list)}')

            valid_alpha_df = pd.read_csv(join(save_dir, 'alpha.csv'))
            self.logger.info(f'alpha.csv existed with length {len(valid_alpha_df)}')
            valid_cdr3a_list = list(set(valid_alpha_df['alpha']) | set(valid_cdr3a_list))
            self.logger.info(f'After adding new alpha data, current length {len(valid_cdr3a_list)}')

            valid_beta_df = pd.read_csv(join(save_dir, 'beta.csv'))
            self.logger.info(f'beta.csv existed with length {len(valid_beta_df)}')
            valid_cdr3b_list = list(set(valid_beta_df['beta']) | set(valid_cdr3b_list))
            self.logger.info(f'After adding new beta data, current length {len(valid_cdr3b_list)}')

        pd.DataFrame({'epitope': list(set(valid_epitope_list))}).to_csv(join(save_dir, 'epitope.csv'), index=False)
        pd.DataFrame({'alpha': list(set(valid_cdr3a_list))}).to_csv(join(save_dir, 'alpha.csv'), index=False)
        pd.DataFrame({'beta': list(set(valid_cdr3b_list))}).to_csv(join(save_dir, 'beta.csv'), index=False)

    def _make_index(self, row, sep='_'):
        return '%s%s%s' % (row['epitope'], sep, row['cdr3b'])

    def VDJdb_loader(self, fn_source):
        self.logger.debug('Loading from %s' % fn_source)
        df = pd.read_table(fn_source, sep='\t', header=0)
        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))

        # Statistics
        self.logger.debug('Unique genes: {}, Unique MHC-class: {}'.format(
            list(set(df['gene'])), list(set(df['mhc.class']))))

        # Select beta CDR3 sequence
        self.logger.debug('Select both alpha beta CDR3 sequences and both MHC-I MHC-II restricted epitopes')
        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))

        # Select valid CDR3 and peptide sequences
        self.logger.debug('Select valid CDR3 and epitope sequences')
        df = df.dropna(subset=['cdr3', 'antigen.epitope'])
        df = df[(df['antigen.epitope'].map(is_valid_aaseq)) & (df['cdr3'].map(is_valid_aaseq))]
        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))

        self.logger.debug('Select confidence score > 0')
        df = df[df['vdjdb.score'].map(lambda score: score > 0)]
        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))

        def _split_alpha_beta(row, target):
            if row['gene'] == target:
                return row['cdr3']
            else:
                return None

        df['epitope'] = df['antigen.epitope'].str.strip().str.upper()
        df['epitope_gene'] = df['antigen.gene']
        df['epitope_species'] = df['antigen.species']
        df['cdr3'] = df['cdr3'].str.strip().str.upper()
        df['cdr3_type'] = df['gene']
        df['mhc_class'] = df['mhc.class']
        df['ref_id'] = df['reference.id']
        df['source'] = 'VDJdb'
        df['cdr3a'] = df.apply(_split_alpha_beta, axis=1, args=(['TRA']))
        df['cdr3b'] = df.apply(_split_alpha_beta, axis=1, args=(['TRB']))

        keep_columns = list(set(df.columns) & set(self.columns))
        df = df.loc[:, keep_columns]

        return df

    def IEDB_Receptor_loader(self, fn_source):
        def _merge(row, col1, col2):
            if str(row[col1]) == 'nan' and str(row[col2]) == 'nan':
                return ''
            return row[col1] if str(row[col1]) != 'nan' else row[col2] 

        self.logger.debug('Loading from %s' % fn_source)
        df = pd.read_csv(fn_source, dtype=str)
        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))

        self.logger.debug('Select T cell (remove B cell).')
        df = df[df['Response Type']=='T cell']
        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))

        df['epitope'] = df['Description'].str.strip().str.upper()
        df['epitope_gene'] = df['Antigen']
        df['epitope_species'] = df['Organism']
        df['cdr3a'] = df.apply(_merge, axis=1, args=('Chain 1 CDR3 Calculated', 'Chain 1 CDR3 Curated')).str.strip().str.upper()
        df['cdr3b'] = df.apply(_merge, axis=1, args=('Chain 2 CDR3 Calculated', 'Chain 2 CDR3 Curated')).str.strip().str.upper()
        df['source'] = 'IEDB-Receptor'
        df['ref_id'] = df['Reference Name'].map(lambda x: 'IEDB:%s' % x)

        self.logger.debug('Select epitope sequences')
        df = df.dropna(subset=['epitope'])
        self.logger.debug('Select valid CDR3 alpha and CDR3 beta sequences.')
        df = df[((df['epitope'].map(is_valid_aaseq)) & (df['cdr3a'].map(is_valid_aaseq))) |
                ((df['epitope'].map(is_valid_aaseq)) & df['cdr3b'].map(is_valid_aaseq))]
        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))

        keep_columns = list(set(df.columns) & set(self.columns))
        df = df.loc[:, keep_columns]
        self.logger.debug('Loaded IEDB Receptor data. Current df_enc.shape: %s' % str(df.shape))
        return df

    def IEDB_Epitope_loader(self, fn_source):
        self.logger.debug('Loading from %s' % fn_source)
        df = pd.read_csv(fn_source, compression='zip', dtype=str, header=1)
        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))

        df['epitope'] = df['Description'].str.strip().str.upper()
        df['epitope_gene'] = df['Antigen Name']
        df['epitope_species'] = df['Organism Name']
        df['cdr3a'] = None
        df['cdr3b'] = None
        df['source'] = 'IEDB-Epitope'

        self.logger.debug('Select epitope sequences')
        df = df.dropna(subset=['epitope'])
        self.logger.debug('Select valid epitope sequences')
        df = df[df['epitope'].map(is_valid_aaseq)]
        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))

        keep_columns = list(set(df.columns) & set(self.columns))
        df = df.loc[:, keep_columns]
        self.logger.debug('Loaded IEDB Epitope data. Current df_enc.shape: %s' % str(df.shape))
        return df

    def GenePlus_Cancer_loader(self, fn_source):
        self.logger.debug('Loading from %s' % fn_source)
        df = pd.read_csv(fn_source, sep='\t', usecols=['aaSeqCDR3'])
        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))

        df['epitope'] = None
        df['cdr3a'] = None
        df['cdr3b'] = df['aaSeqCDR3']

        self.logger.debug('Select valid CDR3 beta sequences')
        df = df.dropna(subset=['cdr3b'])
        df = df[df['cdr3b'].map(is_valid_aaseq)]
        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))

        keep_columns = list(set(df.columns) & set(self.columns))
        df = df.loc[:, keep_columns]
        return df

    def GenePlus_COVID_loader(self, fn_source):
        self.logger.debug('Loading from %s' % fn_source)
        df = pd.read_csv(fn_source, sep='\t', usecols=['aaSeqCDR3'])
        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))

        df['epitope'] = None
        df['cdr3a'] = None
        df['cdr3b'] = df['aaSeqCDR3']

        self.logger.debug('Select valid CDR3 beta sequences')
        df = df.dropna(subset=['cdr3b'])
        df = df[df['cdr3b'].map(is_valid_aaseq)]
        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))

        keep_columns = list(set(df.columns) & set(self.columns))
        df = df.loc[:, keep_columns]
        return df

    def MAB_COVID_loader(self, fn_source):
        self.logger.debug('Loading from %s' % fn_source)
        
        def process_one_file(file_name):
            df = pd.read_csv(join(fn_source, file_name), sep='\t')
            df['cdr3b'] = df['amino_acid']
            df = df[df['cdr3b'] != 'na']
            return df[['cdr3b']]    
        df_list = []
        file_list = listdir(fn_source)
        for file_name in tqdm(file_list):
            file_df = process_one_file(file_name)
            df_list.append(file_df)
        
        df = pd.concat(df_list)
        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))
        self.logger.debug('Select valid CDR3 beta sequences')
        df = df.dropna(subset=['cdr3b'])
        df['epitope'] = None
        df['cdr3a'] = None
        df = df[df['cdr3b'].map(is_valid_aaseq)]
        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))

        keep_columns = list(set(df.columns) & set(self.columns))
        df = df.loc[:, keep_columns]
        return df

    def TCRdb_loader(self, fn_source):
        self.logger.debug('Loading from %s' % fn_source)
        
        def process_one_file(file_name):
            df = pd.read_csv(join(fn_source, file_name), compression='gzip', sep='\t')
            df['cdr3b'] = df['AASeq']
            df['tcrdb.clonefraction'] = df['cloneFraction']
            return df[['cdr3b', 'tcrdb.clonefraction']]    
        df_list = []
        file_list = listdir(fn_source)
        for file_name in tqdm(file_list):
            file_df = process_one_file(file_name)
            df_list.append(file_df)
        
        df = pd.concat(df_list)
        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))
        self.logger.debug('Select valid CDR3 beta sequences')
        df = df.dropna(subset=['cdr3b'])
        df['epitope'] = None
        df['cdr3a'] = None
        df = df[df['cdr3b'].map(is_valid_aaseq)]
        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))

        keep_columns = list(set(df.columns) & set(self.columns))
        df = df.loc[:, keep_columns]
        return df

    def PIRD_loader(self, fn_source):
        self.logger.debug('Loading from %s' % fn_source)
        df = pd.read_csv(fn_source)
        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))
        # Statistics
        self.logger.debug('Unique antigen: {}'.format(len(set(df['Antigen']))))

        df['epitope'] = df['Antigen.sequence'].str.strip().str.upper()
        df['epitope_gene'] = df['Antigen']
        df['epitope_species'] = df['Species']
        df['cdr3a'] = df['CDR3.alpha.aa'].str.strip().str.upper()
        df['cdr3b'] = df['CDR3.beta.aa'].str.strip().str.upper()
        df['ref_id'] = df['Pubmed.id'].map(lambda x: 'Pubmed:%s' % x)

        # Select beta CDR3 sequence
        self.logger.debug('Select both valid alpha beta CDR3 sequences.')
        df = df.dropna(subset=['epitope', 'cdr3a', 'cdr3b'], how='all')
        df = df[(df['epitope'].map(is_valid_aaseq)) | 
                (df['cdr3a'].map(is_valid_aaseq)) |
                (df['cdr3b'].map(is_valid_aaseq))]

        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))

        keep_columns = list(set(df.columns) & set(self.columns))
        df = df.loc[:, keep_columns]
        return df

    def Glanville_loader(self, fn_source):
        self.logger.debug('Loading from %s' % fn_source)
        df = pd.read_csv(fn_source)
        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))
        # Statistics
        self.logger.debug('Unique antigen: {}'.format(list(set(df['Antigen']))))

        df['epitope'] = df['Antigen-peptide'].str.strip().str.upper()
        df['epitope_gene'] = df['Antigen']
        df['epitope_species'] = df['Antigen-species']
        df['cdr3a'] = None
        df['cdr3b'] = df['CDR3b'].str.strip().str.upper()

        # Select beta CDR3 sequence
        self.logger.debug('Select both valid beta CDR3 sequences.')
        df = df.dropna(subset=['epitope', 'cdr3b'], how='all')
        df = df[(df['epitope'].map(is_valid_aaseq)) | 
                (df['cdr3b'].map(is_valid_aaseq))]

        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))

        keep_columns = list(set(df.columns) & set(self.columns))
        df = df.loc[:, keep_columns]
        return df

    def Dash_loader(self, fn_source):
        GENE_INFO_MAP = OrderedDict({
            'BMLF': ('EBV', 'GLCTLVAML', 'HLA-A*02:01'),
            'pp65': ('CMV', 'NLVPMVATV', 'HLA-A*02:01'),
            'M1': ('IAV', 'GILGFVFTL', 'HLA-A*02:01'),
            'F2': ('IAV', 'LSLRNPILV', 'H2-Db'),
            'NP': ('IAV', 'ASNENMETM', 'H2-Db'),
            'PA': ('IAV', 'SSLENFRAYV', 'H2-Db'),
            'PB1': ('IAV', 'SSYRRPVGI', 'H2-Kb'),
            'm139': ('mCMV', 'TVYGFCLL', 'H2-Kb'),
            'M38': ('mCMV', 'SSPPMFRV', 'H2-Kb'),
            'M45': ('mCMV', 'HGIRNASFI', 'H2-Db'),
        })

        self.logger.debug('Loading from %s' % fn_source)
        df = pd.read_csv(fn_source, sep='\t')
        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))

        df['epitope_gene'] = df['epitope']
        df['epitope_species'] = df['epitope_gene'].map(lambda x: GENE_INFO_MAP[x][0])
        df['epitope'] = df['epitope_gene'].map(lambda x: GENE_INFO_MAP[x][1])
        df['mhc'] = df['epitope_gene'].map(lambda x: GENE_INFO_MAP[x][2])
        df['species'] = df['subject'].map(lambda x: 'human' if 'human' in x else 'mouse')
        df['cdr3a'] = df['cdr3a'].str.strip().str.upper()
        df['cdr3b'] = df['cdr3b'].str.strip().str.upper()

        # Statistics
        self.logger.debug('Unique epitope: {}'.format(len(set(df['epitope']))))

        # Select beta CDR3 sequence
        self.logger.debug('Select both valid alpha beta CDR3 sequences.')
        df = df.dropna(subset=['cdr3a', 'cdr3b'], how='all')
        df = df[(df['epitope'].map(is_valid_aaseq)) |
                (df['cdr3a'].map(is_valid_aaseq)) | 
                (df['cdr3b'].map(is_valid_aaseq))]

        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))

        keep_columns = list(set(df.columns) & set(self.columns))
        df = df.loc[:, keep_columns]
        return df

    def McPAS_loader(self, fn_source):
        self.logger.debug('Loading from %s' % fn_source)
        df = pd.read_csv(fn_source, encoding='ISO-8859-1', dtype=str)
        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))

        self.logger.debug('Select valid alpha, beta CDR3 and epitope sequences')
        df = df.dropna(subset=['CDR3.alpha.aa', 'CDR3.beta.aa', 'Epitope.peptide'], how='all')
        df = df[(df['CDR3.beta.aa'].map(is_valid_aaseq)) |
                (df['Epitope.peptide'].map(is_valid_aaseq)) |
                (df['CDR3.alpha.aa'].map(is_valid_aaseq))]
        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))

        df['epitope'] = df['Epitope.peptide'].str.strip().str.upper()
        df['epitope_gene'] = None
        df['epitope_species'] = df['Pathology']
        df['species'] = df['Species']
        df['cdr3a'] = df['CDR3.alpha.aa'].str.strip().str.upper()
        df['cdr3b'] = df['CDR3.beta.aa'].str.strip().str.upper()
        df['mhc'] = df['MHC'].str.strip()
        df['source'] = 'McPAS'
        df['ref_id'] = df['PubMed.ID'].map(lambda x: '%s:%s' % ('PMID', x))

        self.logger.debug('Select human CDR3 sequences.')
        df = df[df['species']=='Human']
        self.logger.debug('Current df.shape: %s' % str(df.shape))

        keep_columns = list(set(df.columns) & set(self.columns))
        df = df.loc[:, keep_columns]
        return df

    def NetTCR_loader(self, fn_source):
        self.logger.debug('Loading from %s' % fn_source)
        df = pd.read_csv(fn_source)
        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))

        df['epitope'] = df['peptide'].str.strip().str.upper()
        df['epitope_gene'] = None
        df['epitope_species'] = None
        df['mhc'] = 'HLA-A*02:01'
        df['cdr3a'] = df['CDR3a'].str.strip().str.upper()
        df['cdr3b'] = df['CDR3b'].str.strip().str.upper()
        df['species'] = 'human'
        df['source'] = 'NetTCR'
        df['ref_id'] = 'PMID:34508155'
        df['label'] = df['binder']

        self.logger.debug('Select valid alpha, beta CDR3 and epitope sequences')
        df = df.dropna(subset=['cdr3a', 'cdr3b', 'epitope'], how='all')
        df = df[(df['cdr3a'].map(is_valid_aaseq)) |
                (df['cdr3b'].map(is_valid_aaseq)) |
                (df['epitope'].map(is_valid_aaseq))]
        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))

        keep_columns = list(set(df.columns) & set(self.columns))
        df = df.loc[:, keep_columns]
        return df

    def huARdb_data_loader(self, fn_source):
        self.logger.debug('Loading from %s' % fn_source)
        df = pd.read_csv(fn_source, usecols=['VJ_chain_CDR3_aa', 'VDJ_chain_CDR3_aa'])
        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))

        df['epitope'] = None
        df['epitope_gene'] = None
        df['epitope_species'] = None
        df['mhc'] = None
        df['cdr3a'] = df['VJ_chain_CDR3_aa'].str.strip().str.upper()
        df['cdr3b'] = df['VDJ_chain_CDR3_aa'].str.strip().str.upper()
        df['species'] = 'human'
        df['source'] = 'ZJU'
        df['ref_id'] = None
        df['label'] = None

        self.logger.debug('Select valid alpha, beta CDR3 and epitope sequences')
        df = df.dropna(subset=['cdr3a', 'cdr3b'], how='all')
        df = df[(df['cdr3a'].map(is_valid_aaseq)) |
                (df['cdr3b'].map(is_valid_aaseq))]
        self.logger.debug('Current df_enc.shape: %s' % str(df.shape))

        keep_columns = list(set(df.columns) & set(self.columns))
        df = df.loc[:, keep_columns]
        return df

