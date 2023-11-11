from config import INTERACTION_MAP_CONFIG
from src.external.bio.peptide_feature import parse_features, parse_operator
from src.external.bio.feature_builder import CombinedPeptideFeatureBuilder


features_list = parse_features(INTERACTION_MAP_CONFIG['features'])
operator = parse_operator(INTERACTION_MAP_CONFIG['operator'])
feature_builder = CombinedPeptideFeatureBuilder(features_list, operator)

print(feature_builder.generate_peptides_feature("CASRGDTFYEQYF", "ATDALMTGF"))