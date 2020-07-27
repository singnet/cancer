import numpy as np
import pandas as pd


class CuratedBreastCancerStudy:
    def __init__(self, patient_idxs, gexs, treatments):
        self.patient_idxs = patient_idxs
        self.gexs = gexs
        self.treatments = treatments
        self.treatments_name2idx = {name: i for i, name in enumerate(
            ['study', 'radio', 'surgery', 'chemo', 'hormone', 'pCR', 'RFS', 'DFS', 'posOutcome']
        )}
        self.po_1_idxs = np.argwhere(self.treatments[:, -1] == 1).reshape(-1)
        self.po_0_idxs = np.argwhere(self.treatments[:, -1] == 0).reshape(-1)

    def get_batch(self, size):
        idxs = np.random.choice(self.patient_idxs.shape[0], size, replace=True)
        return self.patient_idxs[idxs], self.gexs[idxs], self.treatments[idxs]

    def get_balanced_posOutcome_batch(self, size):
        idxs_1 = np.random.choice(
            self.po_1_idxs,
            size // 2 if size % 2 == 0 else (size // 2) + 1,
            replace=True
        )
        idxs_0 = np.random.choice(self.po_0_idxs, size // 2, replace=True)
        ordered_batch_patient_idxs = np.hstack([self.patient_idxs[idxs_1], self.patient_idxs[idxs_0]])
        ordered_batch_gexs = np.vstack([self.gexs[idxs_1], self.gexs[idxs_0]])
        ordered_batch_treatments = np.vstack([self.treatments[idxs_1], self.treatments[idxs_0]])
        permute = np.random.permutation(ordered_batch_patient_idxs.shape[0])
        return ordered_batch_patient_idxs[permute], ordered_batch_gexs[permute], ordered_batch_treatments[permute] #[:, 1:5]

    def get_treatment(self, treatment):
        return self.treatments[:, self.treatments_name2idx[treatment]]

    def __str__(self):
        return '<Study {0} examples>'.format(self.patient_idxs.shape[0])

    def __repr__(self):
        return str(self)


class CuratedBreastCancerData:
    def __init__(self, batch_size, test_split=0.15, study17_file='data/ex15bmcMerged.csv.xz', treatments_file='data/bmc15mldata1.csv'):
        self.batch_size = batch_size
        study17 = pd.read_csv(study17_file)  # load gene expressions data
        treatments = pd.read_csv(treatments_file)  # load treatments data
        # merge gene expressions (studies) and treatments
        self.data = pd.merge(study17, treatments, left_index=True, right_index=True)
        self.data = self.data.fillna(value=-1)  # now NaNs are -1
        # gex data
        self.gex_data = self.data[study17.columns.values[1:]].values
        self.gex_data += np.abs(self.gex_data.min(0))
        self.gex_data /= self.gex_data.max(0)
        self.posOutcome_data = self.data['posOutcome'].values
        # indices for studies
        study_names = list(self.data['study'].unique())  # extract study names
        # mapping study idx -> study name
        self.idxs2study = {idx: study for idx, study in enumerate(study_names)}
        # mapping study name -> study idx
        self.study2idxs = {study: idx for idx, study in enumerate(study_names)}
        self.data = self.data.replace(self.study2idxs)  # convert study name to study id
        self.data = self.data.replace({'mastectomy': 0, 'breast preserving': 1})
        del self.data['patient_ID_y']  # delete patient_ID_y column because it duplicates patient_ID_x
        # divide studies
        self.train_studies = {}
        self.test_studies = {}
        np.random.seed(941211)
        for study_idx in self.idxs2study:
            study_df = self.data.loc[self.data.study == study_idx]  # select particular study
            study_size = study_df.shape[0]
            test_size = int(study_size*test_split)
            test_idxs = np.random.choice(study_df.index, test_size, replace=False)   # choose test items (seed is fixed)
            #print(study_size, test_size, test_idxs, study_df.index)
            test_data = study_df.loc[test_idxs] # study_df.index.isin(
            train_data = study_df.loc[~study_df.index.isin(test_idxs)]
            self.train_studies[study_idx] = CuratedBreastCancerStudy(
                train_data[self.data.columns[0]].values.astype(np.int),  # patient_ID_x
                train_data[self.data.columns[1:-9]].values.astype(np.float),  # gene expression extraction
                train_data[self.data.columns[-9:]].values.astype(np.int)  # treatments
            )
            self.test_studies[study_idx] = CuratedBreastCancerStudy(
                test_data[self.data.columns[0]].values.astype(np.int),  # patient_ID_x
                test_data[self.data.columns[1:-9]].values.astype(np.float),  # gene expression extraction
                test_data[self.data.columns[-9:]].values.astype(np.int)  # treatments
            )
        np.random.seed()
        # epoch size
        study_counts = self.data.study.value_counts()
        self.batches_in_epoch = study_counts.max()*study_counts.shape[0] // self.batch_size
        study_part_condition = self.batch_size // study_counts.shape[0]
        # amount of examples of each study
        self.study_part = study_part_condition if study_part_condition > 0 else 1

    def traverse(self, data, minibatch_size=32):
        for study_idx in data:
            patients, gexs, treatments = data[study_idx] #.get_batch(self.study_part)

    def get_supervised_batch(self):
        for _ in range(self.batches_in_epoch):
            batch_patients, batch_gexs, batch_treatments = [], [], []
            for study_idx in self.train_studies:
                patients, gexs, treatments = self.train_studies[study_idx].get_batch(self.study_part)
                batch_patients.append(patients)
                batch_gexs.append(gexs)
                batch_treatments.append(treatments)
            batch_patients = np.hstack(batch_patients)
            batch_gexs = np.vstack(batch_gexs)
            batch_treatments = np.vstack(batch_treatments)
            shuffle = np.random.permutation(batch_patients.shape[0])
            batch_treatments = batch_treatments[shuffle]
            yield batch_patients[shuffle],\
                  batch_gexs[shuffle],\
                  batch_treatments[:, 0],\
                  batch_treatments[:, 1:]

    def get_unsupervised_batch(self):
        for _, batch_gexs, _, _ in self.get_supervised_batch():
            yield batch_gexs

    def get_batch_bspo(self, part_size, studies):
        study_batch = self.get_supervised_study_posOutcome_batch(part_size, studies)
        batch_gex = []
        batch_study = []
        batch_posOutcome = []
        for study_idx in study_batch:
            gexs, studies, treatments = study_batch[study_idx]
            posOutcomes = treatments[:, -1]
            batch_gex.append(gexs)
            batch_study.append(studies)
            batch_posOutcome.append(posOutcomes)
        batch_gex = np.vstack(batch_gex)
        batch_study = np.hstack(batch_study)
        batch_posOutcome = np.hstack(batch_posOutcome)
        pidx = np.random.permutation(batch_gex.shape[0])
        return batch_gex[pidx], batch_study[pidx], batch_posOutcome[pidx]

    def get_batch_for_siam_study_discrminator(self, part_size, studies):
        study_batch_1 = self.get_supervised_study_posOutcome_batch(part_size, studies)
        rp_study_idxs = np.random.choice(list(study_batch_1.keys()), (4, 2), replace=False)
        # todo: extract positive pairs, then build batch as <e1, e2, y>



    def get_supervised_study_posOutcome_batch(self, part_size, studies):
        batch = {}
        for study_idx in studies:
            patients, gexs, treatments = studies[study_idx].get_balanced_posOutcome_batch(part_size)
            batch[study_idx] = (gexs, treatments[:, 0], treatments[:, 1:])
        return batch
        # +++++++++++++++++++++++++++++++++++++++
        #for _ in range(self.batches_in_epoch):
            #batch_patients, batch_gexs, batch_treatments = [], [], []
            # batch = {}
            # for study_idx in self.train_studies:
            #     patients, gexs, treatments = self.train_studies[study_idx].get_balanced_posOutcome_batch(self.study_part)
            #     batch[study_idx] = (gexs, treatments[:, 0], treatments[:, 1:])
                # batch_patients.append(patients)
                # batch_gexs.append(gexs)
                # batch_treatments.append(treatments)
            #yield batch
            # batch_patients = np.hstack(batch_patients)
            # batch_gexs = np.vstack(batch_gexs)
            # batch_treatments = np.vstack(batch_treatments)
            # shuffle = np.random.permutation(batch_patients.shape[0])
            # batch_treatments = batch_treatments[shuffle]
            # yield batch_patients[shuffle],\
            #       batch_gexs[shuffle],\
            #       batch_treatments[:, 0],\
            #       batch_treatments[:, :]  # 1:5 p_idx, gex, studies, treatments

    # def get_supervised_study_posOutcome_batch(self):
    #     for _ in range(self.batches_in_epoch):
    #         batch_patients, batch_gexs, batch_treatments = [], [], []
    #         for study_idx in self.train_studies:
    #             patients, gexs, treatments = self.train_studies[study_idx].get_balanced_posOutcome_batch(self.study_part)
    #             batch_patients.append(patients)
    #             batch_gexs.append(gexs)
    #             batch_treatments.append(treatments)
    #         batch_patients = np.hstack(batch_patients)
    #         batch_gexs = np.vstack(batch_gexs)
    #         batch_treatments = np.vstack(batch_treatments)
    #         shuffle = np.random.permutation(batch_patients.shape[0])
    #         batch_treatments = batch_treatments[shuffle]
    #         yield batch_patients[shuffle],\
    #               batch_gexs[shuffle],\
    #               batch_treatments[:, 0],\
    #               batch_treatments[:, :]  # 1:5 p_idx, gex, studies, treatments

    def traverse_gex_study(self):
        for i in range(0, self.gex_data.shape[0], self.batch_size):
            batch_gex = self.gex_data[i:i+self.batch_size]
            yield batch_gex


    # def traverse_gex_study(self):
    #     for i in range(0, self.gex_data.shape[0], self.batch_size):
    #         batch_gex = self.gex_data[i:i+self.batch_size]
    #         batch_study = self.study_labels[i:i+self.batch_size]
    #         yield batch_gex, batch_study
    #
    # def get_unsupervised_batch(self):
    #     return self.data.sample(self.batch_size)[self.study17.columns.values[1:]].values
    #
    # def get_study_supervised_batch(self):
    #     df_batch = self.data.sample(self.batch_size)[np.hstack([self.study17.columns.values[1:], ['study']])]
    #     gex = df_batch[self.study17.columns.values[1:]].values
    #     study = [self.study_idxs[s] for s in df_batch['study'].values]
    #     return gex, study



