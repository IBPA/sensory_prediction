import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import scipy.stats


def aggregate_data():
    cof = pd.read_csv('data/cotter_dataset.csv')

    cof_jar = cof[['Flavor.intensity', 'Acidity', 'Mouthfeel']]

    cof_jar_trans = abs(cof_jar-3)
    cof_jar_trans.columns = ['Flavor intensity (adj)', 'Acidity (adj)', 'Mouthfeel (adj)']


    cof_cata = cof.loc[:, 'Tea.floral':]
    cof_cata.drop(columns='Purchase.intent', inplace=True)

    # Dropping Titration pH because it contains nan values and has very low variance anyways
    cof_params = cof.loc[:, 'Dose':'90Sec Temp'].drop(columns=['Setting', 'Grind', 'Titration pH'])
    cof_categorical_params = cof[['Temp.x', 'TDS.x', 'PE.x', 'Setting', 'Grind']]
    cof_other = cof['Purchase.intent']

    cof_categorical_params_brew = pd.concat([cof['Brew'], cof_categorical_params], axis=1).groupby('Brew').first()

    # One hot for NOT grouped
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(cof_categorical_params[['Temp.x', 'TDS.x', 'PE.x', 'Setting', 'Grind']])
    cof_cats_oh = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Temp.x', 'TDS.x', 'PE.x', 'Setting', 'Grind']), index=cof_categorical_params.index)

    # One hot for grouped
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(cof_categorical_params_brew[['Temp.x', 'TDS.x', 'PE.x', 'Setting', 'Grind']])
    cof_cats_brew_oh = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Temp.x', 'TDS.x', 'PE.x', 'Setting', 'Grind']), index=cof_categorical_params_brew.index)

    cof_features = pd.concat([cof_cata, cof_params, cof_cats_oh, cof_jar_trans], axis=1)
    cof_features['liking'] = cof['Liking']
    cof_features['binary_liking'] = (cof['Liking'] >= 6).astype(int)

    cof_features_brew_ = pd.concat([cof['Brew'], cof_cata, cof_params, cof_jar_trans], axis=1)
    cof_features_brew = cof_features_brew_.groupby("Brew").mean()
    cof_features_brew = pd.concat([cof_features_brew, cof_cats_brew_oh], axis=1)
    cof_features_brew['liking'] = cof[['Brew', 'Liking']].groupby('Brew').mean()
    cof_features_brew['binary_liking'] = (cof_features_brew['liking'] >= cof_features_brew['liking'].mean()).astype(int)

    cof_features_judge_ = pd.concat([cof['Judge'], cof_cata, cof_jar_trans], axis=1)
    cof_features_judge = cof_features_judge_.groupby("Judge").mean()
    cof_features_judge['liking'] = cof[['Judge', 'Liking']].groupby('Judge').mean()
    cof_features_judge['binary_liking'] = (cof_features_judge['liking'] >= cof_features_judge['liking'].mean()).astype(int)

    cof_features.to_csv('data/processed/cof_features.csv', index=False)
    cof_features_brew.to_csv('data/processed/cof_features_brew.csv', index=False)
    cof_features_judge.to_csv('data/processed/cof_features_judge.csv', index=False)

def main():
    aggregate_data()

if __name__ == "__main__":
    main()
    


