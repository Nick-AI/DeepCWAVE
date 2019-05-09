import os
import re
import argparse
import numpy as np
import sys
stderr = sys.stderr
# comment out for development
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stderr = open('nul', 'w')
# end comment
import losses
import activations
import pandas as pd
from netCDF4 import Dataset
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Activation, Concatenate, Dropout
from sklearn.externals import joblib
sys.stderr = stderr
from functools import partial, update_wrapper


class WaveHeightRegressor:
    def __init__(self, mdl_weights):
        np.random.seed(13)
        self.mdl_weights = mdl_weights
        self.ann = self._get_network()

    @staticmethod
    def _conv_time(in_t):
        """Converts data acquisition time

        Args:
            in_t: time of data acquisition in format hours since 2010-01-01T00:00:00Z UTC

        Returns:
            Encoding of time where 00:00 and 24:00 are -1 and 12:00 is 1
        """
        in_t = in_t % 24
        return 2 * np.sin((2 * np.pi * in_t) / 48) - 1

    @staticmethod
    def _conv_deg(in_angle, is_inverse=False, in_cos=None, in_sin=None):
        """Converts measurements in degrees (e.g. angles), using encoding proposed at https://stats.stackexchange.com/a/218547
           Encode each angle as tuple theta as tuple (cos(theta), sin(theta)), for justification, see graph at bottom

        Args:
            coord: measurement of lat/ long in degrees

        Returns:
            tuple of values between -1 and 1
        """
        if is_inverse:
            return np.sign(np.rad2deg(np.arcsin(in_sin))) * np.rad2deg(np.arccos(in_cos))

        angle = np.deg2rad(in_angle)
        return (np.cos(angle), np.sin(angle))

    @staticmethod
    def _conv_incAng(in_angle):
        """Converts incidence angle into tuple of values, one angle and one a binary label for wave mode
           Label is 0 if angle is around 23 and 1 if angle is around 37

        Args:
            in_angle: angle, either around 23 or 37 degrees

        Returns:
            tuple of values, one just the angle, the other a binary label
        """
        return (in_angle, int(in_angle > 30))

    def _get_from_ncdf(self, source_file):
        """Reads netcdf file, normalizes some of the data, and returns pandas dataframe

        Args:
            source_file: path to netcdf4 file containing data

        Returns:
            pandas dataframe
        """
        tmp_df = pd.DataFrame()
        ncdf_data = Dataset(source_file, 'r')
        var_keys = ['timeSAR', 'timeALT', 'lonSAR', 'lonALT', 'latSAR', 'latALT', 'hsALT', 'wsALT', 'dx', 'dt', 'nk',
                    'hsSM', 'incidenceAngle', 'sigma0', 'normalizedVariance', 'S', 'hsWW3v2', 'altID']
        time_transf = np.vectorize(self._conv_time)
        coord_transf = np.vectorize(self._conv_deg)
        incAng_transf = np.vectorize(self._conv_incAng)

        for key in var_keys:
            if key == 'S':
                tmp = ncdf_data.variables[key]
                tmp.set_auto_scale(False)
                tmp = tmp[:] * float(tmp.scale_factor)
                for idx in range(20):
                    tmp_df['s' + str(idx)] = tmp[:, idx]
            elif key == 'timeSAR':
                tmp_df['todSAR'] = time_transf(np.array(ncdf_data[key][:]))
            elif key == 'timeALT':
                tmp_df['todALT'] = time_transf(np.array(ncdf_data[key][:]))
            elif key in ['lonSAR', 'lonALT', 'latSAR', 'latALT']:
                cs, sn = coord_transf(np.array(ncdf_data[key][:]))
                tmp_df[key + 'Cos'] = cs
                tmp_df[key + 'Sin'] = sn
            elif key == 'incidenceAngle':
                ang, lbl = incAng_transf(np.array(ncdf_data[key][:]))
                tmp_df['incidenceAngle'] = ang
                tmp_df['incidenceAngleMode'] = lbl
            else:
                tmp_df[key] = ncdf_data[key][:]
        tmp_df['sentinelType'] = int(
            source_file.split('/')[-1].split('_')[0][2] == 'A')  # encodes type A as 1 and B as 0

        # removes all rows with nk!=60
        final_df = tmp_df.loc[tmp_df['nk'] == 60]
        final_df.drop(['nk'], axis=1, inplace=True)

        return final_df

    def _get_from_csv(self, source_file):
        tmp_df = pd.DataFrame()
        src_data = pd.read_csv(source_file)
        var_keys = ['timeSAR', 'timeALT', 'lonSAR', 'lonALT', 'latSAR', 'latALT', 'hsALT', 'wsALT', 'dx', 'dt', 'nk',
                    'hsSM', 'incidenceAngle', 'sigma0', 'normalizedVariance', 'S', 'hsWW3v2', 'altID', 'fileName']
        time_transf = np.vectorize(self._conv_time)
        coord_transf = np.vectorize(self._conv_deg)
        incAng_transf = np.vectorize(self._conv_incAng)
        year_transf = np.vectorize(lambda x: int(re.findall('\d+', x.split('_')[-1])[0][:-2]))
        month_transf = np.vectorize(lambda x: int(re.findall('\d+', x.split('_')[-1])[0][-2:]))
        sentType_transf = np.vectorize(lambda x: int(x.split('/')[-1].split('_')[0][2] == 'A'))

        for key in var_keys:
            if key == 'S':
                funct = np.vectorize(lambda x: np.array(x.split(',')).astype(np.float32), otypes=[np.ndarray])
                transf = np.array([item for item in funct(src_data.loc[:, key].values)], dtype=np.float32)
                for idx in range(20):
                    tmp_df['s' + str(idx)] = transf[:, idx]
            elif key == 'timeSAR':
                tmp_df['todSAR'] = time_transf(src_data.loc[:, key].values)
            elif key == 'timeALT':
                tmp_df['todALT'] = time_transf(src_data.loc[:, key].values)
            elif key in ['lonSAR', 'lonALT', 'latSAR', 'latALT']:
                cs, sn = coord_transf(src_data.loc[:, key].values)
                tmp_df[key + 'Cos'] = cs
                tmp_df[key + 'Sin'] = sn
            elif key == 'incidenceAngle':
                ang, lbl = incAng_transf(src_data.loc[:, key].values)
                tmp_df['incidenceAngle'] = ang
                tmp_df['incidenceAngleMode'] = lbl
            elif key == 'fileName':
                tmp_df['year'] = year_transf(src_data.loc[:, key].values)
                tmp_df['month'] = month_transf(src_data.loc[:, key].values)
                tmp_df['sentinelType'] = sentType_transf(src_data.loc[:, key].values)
            else:
                tmp_df[key] = src_data.loc[:, key].values

        # removes all rows with nk!=60
        final_df = tmp_df.loc[tmp_df['nk'] == 60]
        final_df.drop(['nk'], axis=1, inplace=True)

        return final_df

    def _load_data(self, source_file):
        """Loads netcdf file and replaces some data fields with one-hot vectors

        Args:
            source_file: path to netcdf4 file containing data

        Returns:
            pandas dataframe
        """

        # load data
        if source_file.endswith('.csv'):
            df = self._get_from_csv(source_file)
        elif source_file.endswith('.nc'):
            df = self._get_from_ncdf(source_file)
        else:
            raise TypeError('Invalid file type')

        df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()  # drops all rows with NaNs

        try:
            assert not np.any(np.isnan(df))
        except:
            print('NaN in df')
        try:
            assert np.all(np.isfinite(df))
        except:
            print('Infs in df')

        return df

    def _form_data(self, source_file):
        """Loads netcdf file and preprocesses data for neural network model

        Args:
            source_file: path to netcdf4 file containing data

        Returns:
            pandas dataframe containing fully preprocessed data
        """
        df = self._load_data(source_file)
        norm_cols = ['wsALT', 'dx', 'dt', 'sigma0', 'normalizedVariance']
        norm_cols += ['s' + str(idx) for idx in range(20)]
        for col in norm_cols:
            s_scaler = joblib.load(f'./models/scalerModels/{col}.scl')
            norm_col = np.float64(pd.Series(s_scaler.transform(df[col].values.reshape(-1, 1)).squeeze()))
            df[col] = norm_col

        # Lastly, normalize incidence angle column
        df['idx'] = range(1, len(df)+1)
        md1 = df[df['incidenceAngleMode'] == 0]
        md2 = df[df['incidenceAngleMode'] != 0]
        subsets = [md1.copy(), md2.copy()]

        for idx, s_df in enumerate(subsets):
            s_scaler = joblib.load(f'./models/scalerModels/incidenceAngle{idx}.scl')
            norm_col = np.float64(pd.Series(s_scaler.transform(s_df['incidenceAngle'].values.reshape(-1, 1)).squeeze()))
            s_df['incidenceAngle'] = norm_col
            subsets[idx] = s_df.copy()

        out_df = pd.concat(subsets, ignore_index=True)
        out_df.sort_values(by=['idx'], inplace=True)
        out_df['dt'] = 0
        out_df['dx'] = 0
        out_df['target'] = out_df['hsALT']

        out_df.drop(['idx'], axis=1, inplace=True)

        return out_df

    def _get_network(self):
        """Creates neural network model and loads in pre-trained weights for prediction

        Returns:
            Keras Sequential() instance
        """
        inp = Input(shape=(38,))  # 'hsSM', 'hsWW3v2', 'hsALT', 'altID', 'target' -> dropped
        x = Dense(units=64, activation='relu', kernel_initializer='he_uniform', bias_initializer='zeros')(inp)
        x = Dense(units=64, activation='relu', kernel_initializer='he_uniform', bias_initializer='zeros')(x)
        x = Dense(units=64, activation='relu', kernel_initializer='he_uniform', bias_initializer='zeros')(x)
        x = Dense(units=64, activation='relu', kernel_initializer='he_uniform', bias_initializer='zeros')(x)
        x = Dense(units=64, activation='relu', kernel_initializer='he_uniform', bias_initializer='zeros')(x)
        x = Dense(units=64, activation='relu', kernel_initializer='he_uniform', bias_initializer='zeros')(x)
        x = Dense(units=64, activation='relu', kernel_initializer='he_uniform', bias_initializer='zeros')(x)
        x = Dense(units=64, activation='relu', kernel_initializer='he_uniform', bias_initializer='zeros')(x)
        x = Dense(units=64, activation='relu', kernel_initializer='he_uniform', bias_initializer='zeros')(x)
        x = Dense(units=64, activation='relu', kernel_initializer='he_uniform', bias_initializer='zeros')(x)
        x = Dense(units=64, activation='relu', kernel_initializer='he_uniform', bias_initializer='zeros')(x)
        x = Dense(units=64, activation='relu', kernel_initializer='he_uniform', bias_initializer='zeros')(x)
        x = Dropout(0.5)(x)
        x = Dense(2, activation='linear', kernel_initializer='zeros', bias_initializer='zeros')(x)

        x_mu = Lambda(lambda x: x[:, :1])(x)
        x_sigma = Lambda(lambda x: x[:, 1:])(x)
        sig_act = lambda x: activations.il(x)
        x_sigma = Activation(sig_act)(x_sigma)

        output = Concatenate(axis=-1, name='out')([x_mu, x_sigma])
        net_out = [output]

        mse = partial(losses.Gaussian_MSE)
        update_wrapper(mse, losses.Gaussian_MSE)
        mse.__name__ = 'mean_squared_error'
        metrics = [mse]

        nll = partial(losses.Gaussian_NLL)
        update_wrapper(nll, losses.Gaussian_NLL)
        nll.__name__ = 'negative_log_loss'
        loss = nll

        lr = 0.003
        opt = Adam(lr=lr, clipvalue=1.)

        ann = Model(inputs=inp, outputs=net_out)
        ann.compile(loss=loss, metrics=metrics, optimizer=opt)
        ann.load_weights(self.mdl_weights)
        return ann

    def _get_preds(self, df):
        """Get model predictions for fully pre-processed dataframe

        Args:
            df: fully preprocessed dataframe

        Returns:
            numpy array
        """
        data = df.drop(['hsALT', 'hsWW3v2', 'hsSM', 'altID', 'target'], axis=1).values
        preds = self.ann.predict(data)

        return preds[:, 0]

    def est_hs(self, in_dir):
        """Full data pipeline. Loads data from netcdf -> preprocesses -> gets predictions -> returns values

        Args:
            in_dir: path to netcdf source file

        Returns:
            pandas dataframe with predictions
        """
        data = self._form_data(in_dir)
        predictions = self._get_preds(data)
        out = pd.DataFrame(predictions)
        return out


DESC = 'Produce WaveHeight predictions based on NetCDF4 file(s).'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESC)

    parser.add_argument(
        'input', type=str,
        help='Filename or path to folder with files (cannot contain anything else).\nFile name must have \
             following format: [SARPlatform]_ALT_coloc[yyyy][mm]S.nc'
    )

    parser.add_argument(
        '--outdir', '-o', type=str, required=False, default='./',
        help='Directory (can be absolute path, otherwise it will be created in cwd) for model output csv.'
    )

    parser.add_argument(
        '--weights', '-w', type=str, required=False, default='./models/fullModels/hsALT_regressor.h5',
        help='Model weights file. Defaults to hsALT_regressor.h5 file in ./models/ directory.'
    )

    usr_args = vars(parser.parse_args())

    model = WaveHeightRegressor(usr_args['weights'])

    if usr_args['input'].endswith('.nc') or usr_args['input'].endswith('.csv'):
        print(f'Processing {usr_args["input"]}')
        file = usr_args['input']
        out_file_name = file.split('/')[-1]
        out_file_name = usr_args['outdir'] + ''.join(out_file_name.split('.')[:-1]) + '_preds.csv'
        out = model.est_hs(file)
        out.to_csv(out_file_name)

    else:
        print(f'Processing all files in directory: {usr_args["input"]}')
        if usr_args["input"][-1] is not '/':
            usr_args["input"] += '/'
        files = [usr_args["input"]+f for f in os.listdir(usr_args["input"])]
        for file in files:
            print(f'Processing {file}')
            out_file_name = file.split('/')[-1]
            out_file_name = usr_args['outdir'] + ''.join(out_file_name.split('.')[:-1]) + '_preds.csv'
            out = model.est_hs(file)
            out.to_csv(out_file_name)
