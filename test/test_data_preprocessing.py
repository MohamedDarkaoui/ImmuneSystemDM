import unittest

from src.data.data_processing import *


class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        self.features = 'hydrophob,isoelectric,mass,hydrophil'
        self.operator = 'absdiff'
        self.train_folder = '../data/training_data/'
        self.test_folder = '../data/true_set/'
        self.epitope_list = list_epitopes(self.train_folder)
        self.single_chain = ['TRB_CDR3']
        self.double_chain = ['TRB_CDR3', 'TRA_CDR3']

    def test_load_complete_data(self):
        # single mode
        df = load_complete_data(
            self.epitope_list, self.train_folder, self.single_chain
        )
        self.assertEqual(df.columns.tolist(), ['Epitope', 'TRB_CDR3', 'Label'])

        # combine  mode
        df = load_complete_data(
            self.epitope_list, self.train_folder, self.double_chain
        )
        self.assertEqual(df.columns.tolist(), ['Epitope', 'TRB_CDR3', 'TRA_CDR3', 'Label'])

        # concatenate mode
        df = load_complete_data(
            self.epitope_list, self.train_folder, self.double_chain
        )
        df, chains = concat_columns(df, self.double_chain)
        self.assertEqual(df.columns.tolist(), ['Epitope', 'Label', 'TRB_CDR3TRA_CDR3'])

    def test_calculate_imap_shape(self):
        data = {
            'Epitope': ['ATDALMTGF', 'CINGVCWTV', 'GILGFVFTL'],
            'TRB_CDR3': ['CASSYPYRGLLAGSGNTIYF', 'CASSLEGQLNEQFF', 'CASQSAEASTDTQYF'],
            'TRA_CDR3': ['CAFMSLYGGSQGNLIF', 'CAVVYPLTHGSSNTGKLIF', 'CAENILSSSNTGKLIF'],
            'Label': [1, 0, 0]
        }
        df = pd.DataFrame(data)

        # single mode
        shape = calculate_imap_shape(df, ['TRA_CDR3'], InteractionMapMode.SINGLE)
        self.assertEqual(shape, (19, 9, 4))

        # combine  mode
        shape = calculate_imap_shape(df, ['TRA_CDR3', 'TRB_CDR3'], InteractionMapMode.COMBINE)
        self.assertEqual(shape, (20, 9, 8))

        # concatenate mode
        df, tcr_chains = concat_columns(df, ['TRA_CDR3', 'TRB_CDR3'])
        shape = calculate_imap_shape(df, tcr_chains, InteractionMapMode.CONCATENATE)
        self.assertEqual(shape, (36, 9, 4))

    def test_generate_interaction_maps_single_tcr(self):
        epitope = 'ATDALMTGF'
        tra = 'CAFMSLYGGSQGNLIF'

        chains = [tra]
        imaps = generate_interaction_maps(chains, epitope, self.features, self.operator)

        self.assertEqual(len(imaps), 1)
        self.assertEqual(imaps[0].shape, (len(tra), len(epitope), 4))

    def test_generate_interaction_maps_multiple_tcr(self):
        epitope = 'ATDALMTGF'
        tra = 'CAVVYPLTHGSSNTGKLIF'
        trb = 'CASSLEGQLNEQFF'

        chains = [tra, trb]
        imaps = generate_interaction_maps(chains, epitope, self.features, self.operator)

        self.assertEqual(len(imaps), 2)
        self.assertEqual(imaps[0].shape, (len(tra), len(epitope), 4))
        self.assertEqual(imaps[1].shape, (len(trb), len(epitope), 4))

    def test_generate_imap_dataset_single(self):
        imaps_df, shape = generate_imap_dataset(self.train_folder, self.single_chain, InteractionMapMode.SINGLE)

        self.assertEqual(imaps_df.columns.tolist(), ['interaction_map', 'Label'])
        self.assertEqual(imaps_df.iloc[1, 0].shape, shape)
        self.assertEqual(imaps_df['interaction_map'][5].shape, shape)
        self.assertEqual(shape[2], 4)

    def test_generate_imap_dataset_combine(self):
        imaps_df, shape = generate_imap_dataset(self.train_folder, self.double_chain, InteractionMapMode.COMBINE)

        self.assertEqual(imaps_df.columns.tolist(), ['interaction_map', 'Label'])
        self.assertEqual(imaps_df.iloc[1, 0].shape, shape)
        self.assertEqual(imaps_df['interaction_map'][5].shape, shape)
        self.assertEqual(shape[2], 8)

    def test_generate_imap_dataset_concat(self):
        imaps_df, shape = generate_imap_dataset(
            self.train_folder,
            ['TRA_CDR3', 'TRB_CDR3'],
            InteractionMapMode.CONCATENATE
        )

        self.assertEqual(imaps_df.columns.tolist(), ['interaction_map', 'Label'])
        self.assertEqual(imaps_df.iloc[1, 0].shape, shape)
        self.assertEqual(imaps_df['interaction_map'][5].shape, shape)
        self.assertEqual(shape[2], 4)

    def test_generate_test_data(self):
        test_data = generate_test_data(
            self.test_folder,
            ['TRA_CDR3', 'TRB_CDR3'],
            InteractionMapMode.CONCATENATE,
            (38, 10, 4)
        )
        self.assertEqual(len(test_data[0]), 3)

    def test_imaps(self):
        tra_df, tra_shape = generate_imap_dataset(
            self.train_folder,
            ['TRA_CDR3'],
            InteractionMapMode.SINGLE
        )

        trb_df, trb_shape = generate_imap_dataset(
            self.train_folder,
            ['TRB_CDR3'],
            InteractionMapMode.SINGLE
        )

        # concat_df, concat_shape = generate_imap_dataset(
        #     self.train_folder,
        #     ['TRA_CDR3', 'TRB_CDR3'],
        #     InteractionMapMode.CONCATENATE
        # )

        combine_df, combine_shape = generate_imap_dataset(
            self.train_folder,
            ['TRA_CDR3', 'TRB_CDR3'],
            InteractionMapMode.COMBINE
        )

        imap_a = tra_df['interaction_map'][0]
        imap_b = trb_df['interaction_map'][0]
        # imap_concat = concat_df['interaction_map'][0]
        imap_combine = combine_df['interaction_map'][0]

        shape = 23, 10
        imap_a_padded = pad(imap_a, shape[0], shape[1])
        imap_b_padded = pad(imap_b, shape[0], shape[1])
        self.assertEqual(np.concatenate([imap_a_padded, imap_b_padded], axis=2).tolist(), imap_combine.tolist())


if __name__ == '__main__':
    unittest.main()
