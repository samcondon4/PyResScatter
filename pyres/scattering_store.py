import os
import re
import tempfile
import numpy as np
from numpy.polynomial import Polynomial
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt

from .helpers import circle_fit


class ResonatorScatteringStore(pd.HDFStore):

    def __init__(self, path, geometry, power=True, **kwargs):
        # - Check for legacy flat structure before opening the file permanently - #
        if os.path.exists(path):
            needs_migration = False
            key_mapping = {}
            
            with pd.HDFStore(path, mode='r') as tmp_store:
                keys = tmp_store.keys()
                flat_groups = ['/data', '/meta', '/proc_params']
                for group in flat_groups:
                    if group in keys:
                        needs_migration = True
                        key_mapping[group] = f'/base{group}'
            
            # If flat groups exist, migrate and repack them into /base in one pass
            if needs_migration:
                self._repack(path, key_mapping=key_mapping)
        
        # - Now initialize the parent class as normal - #
        super().__init__(path, mode='a', **kwargs)
        
        assert geometry in ['hanger', 'shunt'], 'Invalid geometry specified. The options are ["hanger", "shunt"]' 
        self.geometry = geometry 
        self.power = power
        self.sparam = '21' if (self.geometry == 'hanger') else '11'
        self.mag_ylabel = r'$|S_{%s}|$ (dB)' % self.sparam 
        self.phase_ylabel = r'$\angle S_{%s}$ (rads)' % self.sparam 
        
        self._index_cache = {}
        
        # - Clean up failed temporary groups from previous run crashes - #
        keys = self.keys()
        for k in keys:
            if k.startswith('/tmp_') or '/tmp_avg_process' in k:
                self.remove(k)

        # - Cache indices for the base data group - #
        if '/base/data' in self.keys() or 'base/data' in self.keys():
            self._get_index_arrays('/base/data')

    # - STATIC METHODS ------------------------------------------------------------------------------------------- #
    @staticmethod
    def _repack(filepath, key_mapping=None):
        """ 
        Repack an HDF5 file to reclaim disk space.
        
        :param filepath: Path to the HDF5 file.
        :type filepath: str
        :param key_mapping: Optional dictionary mapping old keys to new keys to rename groups 
                            during the repacking process without doubling file size.
        :type key_mapping: dict, optional
        """
        if not os.path.exists(filepath):
            return
            
        key_mapping = key_mapping or {}
        
        dir_name = os.path.dirname(os.path.abspath(filepath))
        tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.h5')
        os.close(tmp_fd) 
        
        try:
            with pd.HDFStore(filepath, mode='r') as store_in, pd.HDFStore(tmp_path, mode='w') as store_out:
                for key in store_in.keys():
                    df = store_in.select(key)
                    new_key = key_mapping.get(key, key)
                    store_out.put(new_key, df, format='table')
                    
            os.replace(tmp_path, filepath)
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise e
    
    @staticmethod
    def _compute_color(val, vmin, vmax, cmap='viridis'):
        """ 
        Convert a value between a minimum and maximum to an integer between
        0 and 256 for use in a colormap.
        
        :param val: Integer value to convert to a color.
        :type val: int or float
        :param vmin: Minimum integer value that val can take.
        :type vmin: int or float
        :param vmax: Maximum integer value that val can take.
        :type vmax: int or float
        :param cmap: String identifying the colormap to use.
        :type cmap: str
        """        
        cmap = mpl.colormaps.get_cmap(cmap)
        if vmin == vmax:
            scaled = 0 
        else: 
            scaled = 256 * (val - vmin) / (vmax - vmin)
        if hasattr(val, '__len__'):
            scaled = scaled.astype(int) 
        else:
            scaled = int(scaled)

        return cmap(scaled)

    @staticmethod
    def _line_func(freqs, tau, offset):
        return -2*np.pi*tau*freqs + offset
    
    @staticmethod
    def _configure_subplot_mosaic(mosaic, sweep_param_vals, width_ratios=None, sweep_cmap='viridis', sweep_label=None):
        """ 
        Configure a subplot mosaic and colorbar.

        :param mosaic: List used as input to the subplot_mosaic call.
        :type mosaic: list
        :param sweep_param_vals: Array with the parameter sweep data.
        :type sweep_param_vals: numpy.ndarray
        :param width_ratios: Ratios determining the relative widths of the columns.
        :type width_ratios: list, optional
        :param sweep_cmap: Colormap for the sweep parameter. Defaults to 'viridis'.
        :type sweep_cmap: str, optional
        :param sweep_label: Label for the colorbar.
        :type sweep_label: str, optional
        """
        if sweep_param_vals.shape[0] == 1:
            fig, axs = plt.subplot_mosaic(mosaic) 
        else:
            for row in mosaic:
                row.append('cbar') 
            fig, axs = plt.subplot_mosaic(mosaic, width_ratios=width_ratios) 
            cmap = mpl.colormaps.get_cmap(sweep_cmap)
            sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max() 
            norm = mpl.colors.Normalize(vmin=sweep_min, vmax=sweep_max)
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array(sweep_param_vals)
            fig.colorbar(
                sm, cax=axs['cbar'],
                label='iter' if sweep_label is None else sweep_label
            )
        for key, ax in axs.items():
            if 'iq' in key:
                ax.set_aspect('equal')

        return fig, axs

    @staticmethod 
    def _centered_phase_func(freqs, theta0, Ql, fr):
        return theta0 + 2*np.arctan(2*Ql*(1 - (freqs/fr)))

    def _centered_phase_fit(self, freqs, phase_data, **kwargs):
        params, pcov = spopt.curve_fit(self._centered_phase_func, freqs, phase_data, **kwargs) 

        return params, pcov

    # - INTERNAL HELPERS ---------------------------------------------------------------------------------------- #
    def _get_group_values(self, group, index=None, param=None, index_group=None):
            """ 
            Return the dataframe from the group at the specified index.

            :param group: String corresponding to the hierarchical HDF group to pull the dataframe from.
            :type group: str
            :param index: Index from the RecordGroup and RecordGroupInd list to pull dataframe from.
            :type index: int
            :param param: Parameter to pull from the dataframe. 
            :type param: str, optional
            :param index_group: The baseline data group whose indexing scheme should be used. 
                                If None, defaults to the 'data' group in the same parent directory.
            :type index_group: str, optional
            """
            group = '/' + group.strip('/')
            
            if index_group is None:
                parent_path = group.rsplit('/', 1)[0]
                index_group = f"{parent_path}/data"
            else:
                index_group = '/' + index_group.strip('/')
                
            rg_arr, rgi_arr, rr_arr, start_inds = self._get_index_arrays(index_group)
            if index is not None: 
                ind = start_inds[index]
            else:
                ind = start_inds

            try:
                iter(ind)
            except TypeError:
                rg, rgi = rg_arr[ind], rgi_arr[ind] 
                where_str = f'RecordGroup == "{rg}" & RecordGroupInd == "{rgi}"'
                df = self.select(group, where=where_str) 
            else:
                df = pd.DataFrame() 
                for i in ind:
                    rg, rgi = rg_arr[i], rgi_arr[i] 
                    where_str = f'RecordGroup == "{rg}" & RecordGroupInd == "{rgi}"'
                    df = pd.concat([df, self.select(group, where=where_str)])
                
            if param is not None:
                ret = df[param].values
            else:
                ret = df

            return ret
    
    def _get_index_arrays(self, data_group):
        """ 
        Read and cache the index structures for any data group in the hierarchy. 
        
        :param data_group: String corresponding to the hierarchical path of the data group.
        :type data_group: str
        """
        if not data_group.startswith('/'):
            data_group = '/' + data_group
            
        if data_group not in self._index_cache:
            storer = self.get_storer(data_group)
            rg = storer.read_column('RecordGroup')
            rgi = storer.read_column('RecordGroupInd')
            rr = storer.read_column('RecordRow')
            
            record_start_inds = np.where(rr == '000000')[0]
            self._index_cache[data_group] = (rg, rgi, rr, record_start_inds)
            
        return self._index_cache[data_group]

    # - GENERAL PROCESSING FUNCTIONS ------------------------------------------------------------------ #
    def group_query(self, in_group, query, inds=None):
        """
        Filter plotting `inds` and data points based on a pandas query string.
        Supports mixing trace-level parameters and meta-level scalar parameters.
        
        :param in_group: The group path from which to pull data (e.g., 'base' or 'process_0').
        :type in_group: str
        :param query: Pandas query string using subgroup.column syntax (e.g., 'data.frequency < 5e9').
        :type query: str
        :param inds: Specific trace indices to query over. If None, queries all traces.
        :type inds: numpy.ndarray or list, optional
        :return: A tuple of (valid_inds, point_masks), where valid_inds is an array of 
                 surviving trace indices, and point_masks is a dictionary mapping 
                 the trace ind to an array of valid RecordRow index labels (or None if 
                 only meta-level queries were executed).
        :rtype: tuple(numpy.ndarray, dict or None)
        """
        in_group_clean = '/' + in_group.strip('/')
        data_group = f"{in_group_clean}/data"
        
        rg_arr, rgi_arr, rr_arr, start_inds = self._get_index_arrays(data_group)
        
        if inds is None:
            inds = np.arange(start_inds.shape[0])

        if query is None or not str(query).strip():
            return inds, None
            
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b'
        matches = re.findall(pattern, query)
        
        if not matches:
            return inds, None
            
        subgroups = list(set([m[0] for m in matches]))
        
        trace_dfs = []
        meta_dfs = []
        
        for sg in subgroups:
            path = f"{in_group_clean}/{sg}"
            df_sg = self._get_group_values(path, inds, index_group=data_group).copy()
            
            is_trace = ('RecordRow' in df_sg.index.names) or ('frequency' in df_sg.columns)
            
            df_sg = df_sg.rename(columns={col: f"{sg}_{col}" for col in df_sg.columns})
            
            if is_trace:
                trace_dfs.append(df_sg)
            else:
                meta_dfs.append(df_sg)
        
        meta_combined = None
        if meta_dfs:
            meta_combined = pd.concat(meta_dfs, axis=1)
            meta_combined = meta_combined.loc[:, ~meta_combined.columns.duplicated()]
            
        trace_combined = None
        if trace_dfs:
            trace_combined = pd.concat(trace_dfs, axis=1)
            trace_combined = trace_combined.loc[:, ~trace_combined.columns.duplicated()]
            
        if trace_combined is not None and meta_combined is not None:
            combined_df = trace_combined.join(meta_combined)
        elif trace_combined is not None:
            combined_df = trace_combined
        else:
            combined_df = meta_combined
            
        parsed_query = re.sub(pattern, r'\1_\2', query)
        
        filtered_df = combined_df.query(parsed_query)
        
        if filtered_df.empty:
            return np.array([], dtype=int), None
            
        trace_key_to_ind = {
            (rg_arr[start_inds[i]], rgi_arr[start_inds[i]]): i 
            for i in inds
        }
        
        valid_inds = []
        point_masks = None
        
        if 'RecordRow' in filtered_df.index.names:
            surviving_keys = filtered_df.index.droplevel('RecordRow').unique()
            point_masks = {}
            for (rg, rgi), subset in filtered_df.groupby(level=['RecordGroup', 'RecordGroupInd']):
                if (rg, rgi) in trace_key_to_ind:
                    ind = trace_key_to_ind[(rg, rgi)]
                    point_masks[ind] = subset.index.get_level_values('RecordRow').values
        else:
            surviving_keys = filtered_df.index.unique()
            
        for key in surviving_keys:
            if key in trace_key_to_ind:
                valid_inds.append(trace_key_to_ind[key])
                
        return np.array(sorted(valid_inds)), point_masks
    
    def average_traces_on_sweep(self, in_group, out_group, sweep_param, query=None):
        """ 
        Average traces based on unique values of a given sweep parameter.
        
        :param in_group: The input group path from which to pull data (e.g., 'base').
        :type in_group: str
        :param out_group: The output group path to write the averaged traces into (e.g., 'process_0'). 
                          Supports overwriting if in_group == out_group.
        :type out_group: str
        :param sweep_param: String formatted as 'subgroup.column' to map uniqueness over 
                            (e.g., 'meta.power').
        :type sweep_param: str
        :param query: Optional pandas query string to filter traces prior to averaging. 
                      Point-level masking is not supported for this method.
        :type query: str, optional
        """
        in_group = '/' + in_group.strip('/')
        out_group = '/' + out_group.strip('/')
        data_group = f"{in_group}/data"
        
        subgroup, param = sweep_param.split('.')
        sweep_path = f"{in_group}/{subgroup}"
        
        sweep_df = self.select(sweep_path)
        
        if query is not None:
            inds, point_masks = self.group_query(in_group, query)
            if point_masks is not None and len(point_masks) > 0:
                raise ValueError("Point-level queries (e.g., masking specific frequencies) are not supported for average_traces_on_sweep as averaging requires consistent array shapes.")
                
            rg_arr, rgi_arr, _, start_inds = self._get_index_arrays(data_group)
            valid_tuples = [(rg_arr[start_inds[i]], rgi_arr[start_inds[i]]) for i in inds]
            valid_index = pd.MultiIndex.from_tuples(valid_tuples, names=['RecordGroup', 'RecordGroupInd'])
            sweep_df = sweep_df.loc[sweep_df.index.intersection(valid_index)]

        unique_vals = pd.unique(sweep_df[param])
        
        keys = self.keys()
        meta_keys = [k for k in keys if k.startswith(in_group + '/') and not k.endswith('/data')]
        meta_dfs = {k: self.select(k) for k in meta_keys}
        
        tmp_out_prefix = '/tmp_avg_process'
        
        for k in keys:
            if k.startswith(tmp_out_prefix):
                self.remove(k)
                
        for i, val in enumerate(unique_vals):
            matching_sweep_df = sweep_df[sweep_df[param] == val]
            
            I_list, Q_list = [], []
            freqs = None
            
            for idx in matching_sweep_df.index:
                rg_val, rgi_val = idx[0], idx[1]
                where_str = f'RecordGroup == "{rg_val}" & RecordGroupInd == "{rgi_val}"'
                trace_df = self.select(data_group, where=where_str)
                
                if len(trace_df) == 0:
                    continue
                    
                if freqs is None:
                    freqs = trace_df['frequency'].values
                I_list.append(trace_df['I'].values)
                Q_list.append(trace_df['Q'].values)
                
            if not I_list:
                continue
                
            N = len(I_list)
            I_avg = np.mean(I_list, axis=0)
            Q_avg = np.mean(Q_list, axis=0)
            
            if N > 1:
                I_err = np.std(I_list, axis=0, ddof=1) / np.sqrt(N)
                Q_err = np.std(Q_list, axis=0, ddof=1) / np.sqrt(N)
            else:
                I_err = np.zeros_like(I_avg)
                Q_err = np.zeros_like(Q_avg)
                
            new_rg, new_rgi = '000000', '%06i' % i
            
            data_index = pd.MultiIndex.from_product(
                [[new_rg], [new_rgi], ['%06i' % j for j in range(len(freqs))]],
                names=['RecordGroup', 'RecordGroupInd', 'RecordRow']
            )
            
            avg_df = pd.DataFrame({
                'frequency': freqs, 
                'I': I_avg, 
                'Q': Q_avg,
                'I_err': I_err,
                'Q_err': Q_err
            }, index=data_index)
            self.append(f"{tmp_out_prefix}/data", avg_df)
            
            meta_index = pd.MultiIndex.from_product(
                [[new_rg], [new_rgi]],
                names=['RecordGroup', 'RecordGroupInd']
            )
            
            first_idx = matching_sweep_df.index[0]
            
            for m_key, m_df in meta_dfs.items():
                if first_idx in m_df.index:
                    new_row = m_df.loc[[first_idx]].copy()
                    new_row.index = meta_index
                    
                    if m_key == sweep_path:
                        new_row[param] = val
                        
                    sub_name = m_key.split('/')[-1]
                    self.append(f"{tmp_out_prefix}/{sub_name}", new_row)
                    
        out_keys = [k for k in self.keys() if k.startswith(out_group + '/')]
        for k in out_keys:
            self.remove(k)
            
        tmp_keys = [k for k in self.keys() if k.startswith(tmp_out_prefix)]
        for k in tmp_keys:
            sub_name = k.split('/')[-1]
            df = self.select(k)
            self.append(f"{out_group}/{sub_name}", df)
            self.remove(k)
            
        self._get_index_arrays(f"{out_group}/data")

    def average_res_params_on_sweep(self, in_group, sweep_param, query=None):
        """ 
        Average the fitted resonator parameters based on unique values of a given sweep parameter.
        
        :param in_group: The input group path from which to pull data (e.g., 'base').
        :type in_group: str
        :param sweep_param: String formatted as 'subgroup.column' to map uniqueness over.
        :type sweep_param: str
        :param query: Optional pandas query string to filter parameters prior to averaging.
        :type query: str, optional
        """
        in_group = '/' + in_group.strip('/')
        params_group = f"{in_group}/res_params"
        
        if params_group not in self.keys():
            raise KeyError(f"No res_params found at {params_group}. Please run fit_res_params first.")
            
        subgroup, param = sweep_param.split('.')
        sweep_path = f"{in_group}/{subgroup}"
        
        sweep_df = self.select(sweep_path)
        res_df = self.select(params_group)
        
        if query is not None:
            inds, _ = self.group_query(in_group, query)
            rg_arr, rgi_arr, _, start_inds = self._get_index_arrays(f"{in_group}/data")
            valid_tuples = [(rg_arr[start_inds[i]], rgi_arr[start_inds[i]]) for i in inds]
            valid_index = pd.MultiIndex.from_tuples(valid_tuples, names=['RecordGroup', 'RecordGroupInd'])
            res_df = res_df.loc[res_df.index.intersection(valid_index)]
        
        unique_vals = pd.unique(sweep_df[param])
        
        avg_records = []
        index_tuples = []
        
        for val in unique_vals:
            matching_indices = sweep_df[sweep_df[param] == val].index
            valid_indices = matching_indices.intersection(res_df.index)
            
            if len(valid_indices) == 0:
                continue
                
            subset_df = res_df.loc[valid_indices]
            N = len(subset_df)
            
            row = {}
            
            base_cols = [c for c in subset_df.columns if not c.endswith('_err')]
            
            for col in base_cols:
                vals = subset_df[col].values
                row[col] = np.mean(vals)
                
                sem = np.std(vals, ddof=1) / np.sqrt(N) if N > 1 else 0.0
                
                err_col = f"{col}_err"
                if err_col in subset_df.columns:
                    fit_errs = subset_df[err_col].values
                    prop_err = np.sqrt(np.sum(fit_errs**2)) / N
                    row[err_col] = np.sqrt(sem**2 + prop_err**2)
                else:
                    row[err_col] = sem
            
            sweep_vals = sweep_df.loc[valid_indices, param].values
            row[param] = np.mean(sweep_vals)
            
            sweep_sem = np.std(sweep_vals, ddof=1) / np.sqrt(N) if N > 1 else 0.0
            
            sweep_err_col = f"{param}_err"
            if sweep_err_col in sweep_df.columns:
                sweep_fit_errs = sweep_df.loc[valid_indices, sweep_err_col].values
                sweep_prop_err = np.sqrt(np.sum(sweep_fit_errs**2)) / N
                row[sweep_err_col] = np.sqrt(sweep_sem**2 + sweep_prop_err**2)
            else:
                row[sweep_err_col] = sweep_sem
                    
            avg_records.append(row)
            
            index_tuples.append(valid_indices[0])
            
        if avg_records:
            avg_res_df = pd.DataFrame(avg_records)
            avg_res_df.index = pd.MultiIndex.from_tuples(index_tuples, names=['RecordGroup', 'RecordGroupInd'])
            
            out_path = f"{in_group}/avg_res_params"
            if out_path in self.keys():
                self.remove(out_path)
                
            self.append(out_path, avg_res_df)

    # - CALIBRATION FUNCTIONS -------------------------------------------------------------------------- #
    def calibrate_cable_delay(self, 
                in_group, out_group,
                tau=None, offset=None, 
                fit_frequency_bound=None, inds=None, query=None,
                plot=False, sweep_param=None, sweep_cmap='viridis', sweep_label=None,
            ):
            """ 
            Remove a line from the unwrapped phase data.
            
            :param in_group: The group path from which to pull data (e.g., 'base').
            :type in_group: str
            :param out_group: The output group path to write the calibrated data into (e.g., 'cal_cable').
                              Supports overwriting if in_group == out_group.
            :type out_group: str
            :param tau: Fixed cable delay slope. If None a line will be fit to the unwrapped phase.
            :type tau: float, optional
            :param offset: Fixed cable delay offset.
            :type offset: float, optional
            :param fit_frequency_bound: Frequency range over which a line fit should be performed. 
            :type fit_frequency_bound: tuple, optional
            :param inds: Indices over which to perform the calibration. 
            :type inds: list or numpy.ndarray, optional
            :param query: Optional pandas query string to filter traces and points.
            :type query: str, optional
            :param plot: Boolean to indicate if a plot showing the calibration results should be generated. 
            :type plot: bool, optional
            :param sweep_param: String to indicate a parameter that is swept over in the data.
            :type sweep_param: str, optional
            :param sweep_cmap: Colormap to use to indicate the value of the swept parameter.
            :type sweep_cmap: str, optional
            :param sweep_label: String label used to label the colorbar. 
            :type sweep_label: str, optional
            """
            in_group = '/' + in_group.strip('/')
            out_group = '/' + out_group.strip('/')
            data_group = f"{in_group}/data"
            tmp_out_prefix = '/tmp_cal_process'
            
            keys = self.keys()
            for k in keys:
                if k.startswith(tmp_out_prefix):
                    self.remove(k)

            rg_arr, rgi_arr, rr_arr, start_inds = self._get_index_arrays(data_group)

            if inds is None:
                inds = np.arange(start_inds.shape[0])
                
            point_masks = None
            if query is not None:
                inds, point_masks = self.group_query(in_group, query, inds)

            if sweep_param is None:
                sweep_param_vals = np.arange(start_inds.shape[0])[inds] if len(inds) > 0 else np.array([])
                param = 'iter' 
            else:
                subgroup, param = sweep_param.split('.') 
                sweep_path = f"{in_group}/{subgroup}"
                sweep_param_vals = self[sweep_path][param].values[inds]
                
            if len(sweep_param_vals) > 0:
                sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max()
            else:
                sweep_min, sweep_max = 0, 1
            
            if plot:
                fig, axs = self._configure_subplot_mosaic(
                    [['phase_raw', 'iq_raw'], ['phase_cal', 'iq_cal']],
                    sweep_param_vals,
                    width_ratios=[0.475, 0.475, 0.05],
                    sweep_label=sweep_label,
                    sweep_cmap=sweep_cmap,
                )
                axs['phase_cal'].set(xlabel='Frequency (GHz)', ylabel=self.phase_ylabel)
                axs['phase_raw'].set(xlabel='Frequency (GHz)', ylabel=self.phase_ylabel)
                axs['iq_cal'].set(ylabel='Q', xlabel='I')
                axs['iq_raw'].set(ylabel='Q', xlabel='I')
                ret = fig, axs
            else:
                ret = None

            try: 
                target_indices = [(rg_arr[start_inds[i]], rgi_arr[start_inds[i]]) for i in inds]
                meta_keys = [k for k in keys if k.startswith(in_group + '/') and not k.endswith('/data')]
                for m_key in meta_keys:
                    m_df = self.select(m_key)
                    subset_df = m_df.loc[m_df.index.isin(target_indices)]
                    sub_name = m_key.split('/')[-1]
                    self.put(f"{tmp_out_prefix}/{sub_name}", subset_df, format='table')
                
                for i, val in zip(inds, sweep_param_vals): 
                    ind = start_inds[i] 
                    rg_val, rgi_val = rg_arr[ind], rgi_arr[ind] 

                    data = self._get_group_values(data_group, i) 
                    
                    if point_masks is not None and i in point_masks:
                        valid_rows = point_masks[i]
                        if 'RecordRow' in data.index.names:
                            mask = data.index.get_level_values('RecordRow').isin(valid_rows)
                            data = data.loc[mask]
                            
                    if data.empty:
                        continue
                        
                    freqs = data.frequency.values  
                    I, Q = data.I.values, data.Q.values
                    phase = np.unwrap(np.arctan2(Q, I))
                    mlin = np.sqrt(I**2 + Q**2)
                    
                    if fit_frequency_bound is not None:
                        fit_inds_arr = (fit_frequency_bound[0] < freqs) * (freqs < fit_frequency_bound[1])
                        fit_freqs = freqs[fit_inds_arr]
                        fit_phase = phase[fit_inds_arr]
                    else:
                        fit_freqs = freqs
                        fit_phase = phase
                        
                    if (tau is None) and (offset is None):
                        fit_func = self._line_func
                        popt, pcov = spopt.curve_fit(fit_func, fit_freqs, fit_phase)
                        tau_fit, offset_fit = popt
                    elif (tau is None) and (offset is not None):
                        fit_func = lambda f, t: self._line_func(f, t, offset)
                        popt, pcov = spopt.curve_fit(fit_func, fit_freqs, fit_phase)
                        tau_fit = popt[0]
                        offset_fit = offset 
                    elif (tau is not None) and (offset is None):
                        fit_func = lambda f, off: self._line_func(f, tau, off)
                        popt, pcov = spopt.curve_fit(fit_func, fit_freqs, fit_phase)
                        tau_fit = tau 
                        offset_fit = popt[0] 
                    else:
                        tau_fit = tau
                        offset_fit = offset 
                        
                    line = self._line_func(freqs, tau_fit, offset_fit)
                    corrected_phase = phase - line
                    Ical, Qcal = mlin*np.cos(corrected_phase), mlin*np.sin(corrected_phase)
                    
                    cal_df = pd.DataFrame(
                        {'frequency': freqs, 'I': Ical, 'Q': Qcal}, 
                        index=pd.MultiIndex.from_product(
                            [[rg_val], [rgi_val], ['%06i' % j for j in np.arange(freqs.shape[0])]],
                            names=['RecordGroup', 'RecordGroupInd', 'RecordRow'] 
                        )
                    )
                    self.append(f"{tmp_out_prefix}/data", cal_df) 
                    
                    params_df = pd.DataFrame(
                        {'tau': tau_fit, 'cable_delay_offset': offset_fit}, 
                        index=pd.MultiIndex.from_product(
                            [[rg_val], [rgi_val]], names=['RecordGroup', 'RecordGroupInd']
                        )
                    ) 
                    self.append(f"{tmp_out_prefix}/cable_delay_params", params_df) 
                    
                    if plot:
                        plot_freqs = freqs*1e-9
                        color = self._compute_color(val, sweep_min, sweep_max, sweep_cmap) 
                        axs['phase_raw'].plot(plot_freqs, phase, color=color)
                        axs['phase_raw'].plot(plot_freqs, line, ls=':', color='black')
                        axs['phase_cal'].plot(plot_freqs, corrected_phase, color=color)
                        axs['iq_raw'].scatter(I, Q, color=color, marker='.')
                        axs['iq_cal'].scatter(Ical, Qcal, color=color, marker='.')

                out_keys = [k for k in self.keys() if k.startswith(out_group + '/')]
                for k in out_keys:
                    self.remove(k)
                    
                tmp_keys = [k for k in self.keys() if k.startswith(tmp_out_prefix)]
                for k in tmp_keys:
                    sub_name = k.split('/')[-1]
                    df = self.select(k)
                    self.append(f"{out_group}/{sub_name}", df)
                    self.remove(k)
                    
                self._get_index_arrays(f"{out_group}/data")
            
            except Exception as e:
                for k in self.keys():
                    if k.startswith(tmp_out_prefix):
                        self.remove(k)
                raise e

            return ret
    
    def calibrate_constant_scaling(self,
                in_group, out_group,
                a=None, alpha=None, phase_fit_kwargs=None,
                inds=None, query=None, plot=False,
                sweep_param=None, sweep_cmap='viridis', sweep_label=None, 
            ):
            """
            Remove constant magnitude and phase offset scaling factors from the data.
            
            :param in_group: The group path from which to pull data.
            :type in_group: str
            :param out_group: The output group path for the calibrated data.
            :type out_group: str
            :param a: Fixed magnitude scaling factor. If None, it will be fitted.
            :type a: float, optional
            :param alpha: Fixed phase offset. If None, it will be fitted.
            :type alpha: float, optional
            :param phase_fit_kwargs: Keyword arguments for the internal centered phase fit.
            :type phase_fit_kwargs: dict, optional
            :param inds: Indices over which to perform the calibration.
            :type inds: list or numpy.ndarray, optional
            :param query: Optional pandas query string to filter traces and points.
            :type query: str, optional
            :param plot: Boolean to indicate if a plot should be generated.
            :type plot: bool, optional
            :param sweep_param: String to indicate a parameter that is swept over.
            :type sweep_param: str, optional
            :param sweep_cmap: Colormap to use.
            :type sweep_cmap: str, optional
            :param sweep_label: String label used to label the colorbar.
            :type sweep_label: str, optional
            """
            in_group = '/' + in_group.strip('/')
            out_group = '/' + out_group.strip('/')
            data_group = f"{in_group}/data"
            tmp_out_prefix = '/tmp_cal_process'
            
            keys = self.keys()
            for k in keys:
                if k.startswith(tmp_out_prefix):
                    self.remove(k)

            rg_arr, rgi_arr, rr_arr, start_inds = self._get_index_arrays(data_group)

            if inds is None:
                inds = np.arange(start_inds.shape[0])
                
            point_masks = None
            if query is not None:
                inds, point_masks = self.group_query(in_group, query, inds)
                
            if sweep_param is None:
                sweep_param_vals = np.arange(start_inds.shape[0])[inds] if len(inds) > 0 else np.array([])
                param = 'iter' 
            else:
                subgroup, param = sweep_param.split('.') 
                sweep_path = f"{in_group}/{subgroup}"
                sweep_param_vals = self[sweep_path][param].values[inds]
                
            if len(sweep_param_vals) > 0:
                sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max() 
            else:
                sweep_min, sweep_max = 0, 1

            if plot:
                fig, axs = self._configure_subplot_mosaic(
                    [['iq_raw', 'iq_process'], ['centered_phase', 'iq_final']],
                    sweep_param_vals=sweep_param_vals, width_ratios=[0.475, 0.475, 0.05],
                    sweep_label=sweep_label, sweep_cmap=sweep_cmap,
                )
                for key, ax in axs.items():
                    if 'iq' in key:
                        ax.set(xlabel='I', ylabel='Q')
                axs['centered_phase'].set(xlabel='Frequency (GHz)', ylabel=self.phase_ylabel) 
                ret = fig, axs
            else:
                ret = None

            try: 
                target_indices = [(rg_arr[start_inds[i]], rgi_arr[start_inds[i]]) for i in inds]
                meta_keys = [k for k in keys if k.startswith(in_group + '/') and not k.endswith('/data')]
                for m_key in meta_keys:
                    m_df = self.select(m_key)
                    subset_df = m_df.loc[m_df.index.isin(target_indices)]
                    sub_name = m_key.split('/')[-1]
                    self.put(f"{tmp_out_prefix}/{sub_name}", subset_df, format='table')
                    
                for i, val in zip(inds, sweep_param_vals):
                    ind = start_inds[i]
                    rg_val, rgi_val = rg_arr[ind], rgi_arr[ind]
                    
                    data = self._get_group_values(data_group, i)
                    
                    if point_masks is not None and i in point_masks:
                        valid_rows = point_masks[i]
                        if 'RecordRow' in data.index.names:
                            mask = data.index.get_level_values('RecordRow').isin(valid_rows)
                            data = data.loc[mask]
                            
                    if data.empty:
                        continue
                        
                    I, Q, freqs = data.I.values, data.Q.values, data.frequency.values
                    mlin = np.sqrt(I**2 + Q**2) 
                    phase = np.unwrap(np.arctan2(Q, I)) 
                    sdata = mlin*np.exp(1j*phase)

                    if plot:
                        color = self._compute_color(val, sweep_min, sweep_max, sweep_cmap)
                        plot_freqs = freqs*1e-9 
                        axs['iq_raw'].scatter(I, Q, marker='.', color=color)

                    if (a is None) or (alpha is None): 
                        xc, yc, r = circle_fit(sdata)
                        Icentered = I - xc
                        Qcentered = Q - yc
                        centered_phase = np.unwrap(np.arctan2(Qcentered, Icentered))

                        phase_fit_kwargs = {} if phase_fit_kwargs is None else phase_fit_kwargs 
                        params, pcov = self._centered_phase_fit(freqs, centered_phase, **phase_fit_kwargs)
                        theta0, Ql, fr = params

                        beta = (theta0 + np.pi)
                        offres = xc + r*np.cos(beta) + 1j*(yc + r*np.sin(beta))
                        afit, alphafit = np.abs(offres), np.arctan2(np.imag(offres), np.real(offres))
                        
                        if plot:
                            axs['iq_process'].scatter(I, Q, color=color, marker='.') 
                            axs['iq_process'].scatter(Icentered, Qcentered, color=color, marker='.') 
                            axs['iq_process'].plot([xc, np.real(offres)], [yc, np.imag(offres)], color='black', marker='o')
                            axs['centered_phase'].plot(plot_freqs, centered_phase, color=color)
                            phase_fit = self._centered_phase_func(freqs, theta0, Ql, fr)
                            axs['centered_phase'].plot(plot_freqs, phase_fit, ls=':', color='black')

                    a_cal = afit if a is None else a
                    alpha_cal = alphafit if alpha is None else alpha
                    factor = a_cal*np.exp(1j*alpha_cal)
                    sdata /= factor
                    cal_I, cal_Q = np.real(sdata), np.imag(sdata) 

                    if plot:
                        axs['iq_final'].scatter(cal_I, cal_Q, marker='.', color=color)

                    cal_df = pd.DataFrame({'frequency': freqs, 'I': cal_I, 'Q': cal_Q},
                        index=pd.MultiIndex.from_product(
                        [[rg_val], [rgi_val], ['%06i' % j for j in np.arange(freqs.shape[0])]],
                        names=['RecordGroup', 'RecordGroupInd', 'RecordRow'] 
                    ))
                    self.append(f"{tmp_out_prefix}/data", cal_df) 
                    
                    cal_params_df = pd.DataFrame({'a': a_cal, 'alpha': alpha_cal}, 
                        index=pd.MultiIndex.from_product(
                        [[rg_val], [rgi_val]], names=['RecordGroup', 'RecordGroupInd']
                    ))
                    self.append(f"{tmp_out_prefix}/constant_scaling_params", cal_params_df)

                out_keys = [k for k in self.keys() if k.startswith(out_group + '/')]
                for k in out_keys:
                    self.remove(k)
                    
                tmp_keys = [k for k in self.keys() if k.startswith(tmp_out_prefix)]
                for k in tmp_keys:
                    sub_name = k.split('/')[-1]
                    df = self.select(k)
                    self.append(f"{out_group}/{sub_name}", df)
                    self.remove(k)
                    
                self._get_index_arrays(f"{out_group}/data")

            except Exception as e:
                for k in self.keys():
                    if k.startswith(tmp_out_prefix): self.remove(k)
                raise e
            return ret

    def calibrate_polymag_background(self,
                in_group, out_group,
                lower_frequency_bound=None, upper_frequency_bound=None, 
                inds=None, query=None, degree=2, fixed_coeffs=None, domain=None,
                plot=False, sweep_param=None, sweep_cmap='viridis', sweep_label=None, 
            ):
            """
            Calibrate magnitude background by fitting a polynomial to the non-resonant regions.
            
            :param in_group: The group path from which to pull data.
            :type in_group: str
            :param out_group: The output group path to write calibrated data into.
            :type out_group: str
            :param lower_frequency_bound: Lower region to fit the polynomial over.
            :type lower_frequency_bound: tuple, optional
            :param upper_frequency_bound: Upper region to fit the polynomial over.
            :type upper_frequency_bound: tuple, optional
            :param inds: Indices over which to perform the calibration.
            :type inds: list or numpy.ndarray, optional
            :param query: Optional pandas query string to filter traces and points.
            :type query: str, optional
            :param degree: Polynomial degree to fit. Defaults to 2.
            :type degree: int, optional
            :param fixed_coeffs: Fixed coefficients for the polynomial if skipping the fit.
            :type fixed_coeffs: array-like, optional
            :param domain: Evaluation domain for the polynomial if providing fixed coefficients.
            :type domain: tuple, optional
            :param plot: Generate a plot of the calibration.
            :type plot: bool, optional
            :param sweep_param: String to indicate a parameter that is swept over.
            :type sweep_param: str, optional
            :param sweep_cmap: Colormap to use.
            :type sweep_cmap: str, optional
            :param sweep_label: String label used to label the colorbar.
            :type sweep_label: str, optional
            """
            in_group = '/' + in_group.strip('/')
            out_group = '/' + out_group.strip('/')
            data_group = f"{in_group}/data"
            tmp_out_prefix = '/tmp_cal_process'
            
            keys = self.keys()
            for k in keys:
                if k.startswith(tmp_out_prefix): self.remove(k)

            rg_arr, rgi_arr, rr_arr, start_inds = self._get_index_arrays(data_group)

            if inds is None:
                inds = np.arange(start_inds.shape[0])
                
            point_masks = None
            if query is not None:
                inds, point_masks = self.group_query(in_group, query, inds)
                
            if sweep_param is None:
                sweep_param_vals = np.arange(start_inds.shape[0])[inds] if len(inds) > 0 else np.array([])
                param = 'iter' 
            else:
                subgroup, param = sweep_param.split('.') 
                sweep_path = f"{in_group}/{subgroup}"
                sweep_param_vals = self[sweep_path][param].values[inds]
                
            if len(sweep_param_vals) > 0:
                sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max() 
            else:
                sweep_min, sweep_max = 0, 1

            if plot:
                fig, axs = self._configure_subplot_mosaic(
                    [['mag_raw'], ['mag_cal']], sweep_param_vals, width_ratios=[0.95, 0.05],
                    sweep_label=sweep_label, sweep_cmap=sweep_cmap,
                )
                axs['mag_raw'].set(xlabel='Frequency (GHz)', ylabel=self.mag_ylabel)
                axs['mag_cal'].set(xlabel='Frequency (GHz)', ylabel=self.mag_ylabel)
                ret = fig, axs
            else:
                ret = None

            try: 
                target_indices = [(rg_arr[start_inds[i]], rgi_arr[start_inds[i]]) for i in inds]
                meta_keys = [k for k in keys if k.startswith(in_group + '/') and not k.endswith('/data')]
                for m_key in meta_keys:
                    m_df = self.select(m_key)
                    subset_df = m_df.loc[m_df.index.isin(target_indices)]
                    sub_name = m_key.split('/')[-1]
                    self.put(f"{tmp_out_prefix}/{sub_name}", subset_df, format='table')

                for i, val in zip(inds, sweep_param_vals):
                    ind = start_inds[i]
                    rg_val, rgi_val = rg_arr[ind], rgi_arr[ind] 
                    
                    data = self._get_group_values(data_group, i)
                    
                    if point_masks is not None and i in point_masks:
                        valid_rows = point_masks[i]
                        if 'RecordRow' in data.index.names:
                            mask = data.index.get_level_values('RecordRow').isin(valid_rows)
                            data = data.loc[mask]
                            
                    if data.empty:
                        continue
                        
                    I, Q, freqs = data.I.values, data.Q.values, data.frequency.values
                    mlog = (1 + 1*self.power)*10*np.log10(np.sqrt(I**2 + Q**2))
                    phase = np.unwrap(np.arctan2(Q, I)) 
                    
                    if lower_frequency_bound is not None:
                        lower_inds = np.where((lower_frequency_bound[0] <= freqs) * (freqs <= lower_frequency_bound[1]))[0] 
                        lower_I, lower_Q, lower_freqs = I[lower_inds], Q[lower_inds], freqs[lower_inds]
                        lower_mlog = (1 + 1*self.power)*10*np.log10(np.sqrt(lower_I**2 + lower_Q**2))
                    if upper_frequency_bound is not None:
                        upper_inds = np.where((upper_frequency_bound[0] <= freqs) * (freqs <= upper_frequency_bound[1]))[0] 
                        upper_I, upper_Q, upper_freqs = I[upper_inds], Q[upper_inds], freqs[upper_inds]
                        upper_mlog = (1 + 1*self.power)*10*np.log10(np.sqrt(upper_I**2 + upper_Q**2))
                        
                    if fixed_coeffs is None:
                        if (upper_frequency_bound is None) and (lower_frequency_bound is not None):
                            fit = Polynomial.fit(lower_freqs, lower_mlog, degree)
                            fit = (lower_freqs, fit) 
                        elif (upper_frequency_bound is not None) and (lower_frequency_bound is None):
                            fit = Polynomial.fit(upper_freqs, upper_mlog, degree)
                            fit = (upper_freqs, fit) 
                        elif (upper_frequency_bound is not None) and (lower_frequency_bound is not None):
                            fit = Polynomial.fit(np.concatenate([lower_freqs, upper_freqs]), np.concatenate([lower_mlog, upper_mlog]), degree) 
                            fit = (freqs, fit) 
                        else: 
                            fit = (freqs, Polynomial.fit(freqs, mlog, degree))
                    elif domain is not None:
                        fit = Polynomial(fixed_coeffs, domain=domain) 
                        if (upper_frequency_bound is None) and (lower_frequency_bound is not None):
                            fit = (lower_freqs, fit) 
                        elif (upper_frequency_bound is not None) and (lower_frequency_bound is None):
                            fit = (upper_freqs, fit) 
                        else:
                            fit = (freqs, fit) 
                    else:
                        raise ValueError('domain must be provided for a fixed polynomial fit.')

                    fit_mlog = np.zeros_like(freqs)
                    fit_mlog[(fit[0].min() <= freqs) * (freqs <= fit[0].max())] = fit[1](fit[0]) 
                    cal_mlog = mlog - fit_mlog
                    cal_mlin = 10**(cal_mlog / ((1 + 1*self.power)*10))
                    cal_I, cal_Q = cal_mlin*np.cos(phase), cal_mlin*np.sin(phase)

                    cal_df = pd.DataFrame({'frequency': freqs, 'I': cal_I, 'Q': cal_Q},
                        index=pd.MultiIndex.from_product(
                        [[rg_val], [rgi_val], ['%06i' % j for j in np.arange(freqs.shape[0])]],
                        names=['RecordGroup', 'RecordGroupInd', 'RecordRow'] 
                    ))
                    self.append(f"{tmp_out_prefix}/data", cal_df)
                    
                    cal_params_dict = {f'x{j}': fit[1].coef[j] for j in range(fit[1].coef.shape[0])} 
                    cal_params_dict['domain_min'] = fit[1].domain.min()
                    cal_params_dict['domain_max'] = fit[1].domain.max() 
                    cal_params_df = pd.DataFrame(cal_params_dict, index=pd.MultiIndex.from_product(
                        [[rg_val], [rgi_val]], names=['RecordGroup', 'RecordGroupInd'] 
                    ))
                    self.append(f"{tmp_out_prefix}/polymag_params", cal_params_df)

                    if plot:
                        plot_freqs = freqs*1e-9 
                        color = self._compute_color(val, sweep_min, sweep_max, sweep_cmap) 
                        axs['mag_raw'].plot(plot_freqs, mlog, color=color)
                        axs['mag_raw'].plot(plot_freqs, fit_mlog, ls=':', color='black') 
                        axs['mag_cal'].plot(plot_freqs, cal_mlog, color=color)

                out_keys = [k for k in self.keys() if k.startswith(out_group + '/')]
                for k in out_keys: self.remove(k)
                    
                tmp_keys = [k for k in self.keys() if k.startswith(tmp_out_prefix)]
                for k in tmp_keys:
                    sub_name = k.split('/')[-1]
                    df = self.select(k)
                    self.append(f"{out_group}/{sub_name}", df)
                    self.remove(k)
                    
                self._get_index_arrays(f"{out_group}/data")
            
            except Exception as e:
                for k in self.keys():
                    if k.startswith(tmp_out_prefix): self.remove(k)
                raise e
            return ret

    def calibrate_polyphase_background(self,
            in_group, out_group,
            lower_frequency_bound=None, upper_frequency_bound=None,
            inds=None, query=None, degree=2, fixed_coeffs=None, domain=None, plot=False, 
            sweep_param=None, sweep_cmap='viridis', sweep_label=None,
        ):
        """
        Calibrate phase background by fitting a polynomial to the non-resonant regions.
        
        :param in_group: The group path from which to pull data.
        :type in_group: str
        :param out_group: The output group path to write calibrated data into.
        :type out_group: str
        :param lower_frequency_bound: Lower region to fit the polynomial over.
        :type lower_frequency_bound: tuple, optional
        :param upper_frequency_bound: Upper region to fit the polynomial over.
        :type upper_frequency_bound: tuple, optional
        :param inds: Indices over which to perform the calibration.
        :type inds: list or numpy.ndarray, optional
        :param query: Optional pandas query string to filter traces and points.
        :type query: str, optional
        :param degree: Polynomial degree to fit. Defaults to 2.
        :type degree: int, optional
        :param fixed_coeffs: Fixed coefficients for the polynomial if skipping the fit.
        :type fixed_coeffs: array-like, optional
        :param domain: Evaluation domain for the polynomial if providing fixed coefficients.
        :type domain: tuple, optional
        :param plot: Generate a plot of the calibration.
        :type plot: bool, optional
        :param sweep_param: String to indicate a parameter that is swept over.
        :type sweep_param: str, optional
        :param sweep_cmap: Colormap to use.
        :type sweep_cmap: str, optional
        :param sweep_label: String label used to label the colorbar.
        :type sweep_label: str, optional
        """
        in_group = '/' + in_group.strip('/')
        out_group = '/' + out_group.strip('/')
        data_group = f"{in_group}/data"
        tmp_out_prefix = '/tmp_cal_process'
        
        keys = self.keys()
        for k in keys:
            if k.startswith(tmp_out_prefix): self.remove(k)

        rg_arr, rgi_arr, rr_arr, start_inds = self._get_index_arrays(data_group)

        if inds is None:
            inds = np.arange(start_inds.shape[0])
            
        point_masks = None
        if query is not None:
            inds, point_masks = self.group_query(in_group, query, inds)
            
        if sweep_param is None:
            sweep_param_vals = np.arange(start_inds.shape[0])[inds] if len(inds) > 0 else np.array([])
            param = 'iter' 
        else:
            subgroup, param = sweep_param.split('.') 
            sweep_path = f"{in_group}/{subgroup}"
            sweep_param_vals = self[sweep_path][param].values[inds]
            
        if len(sweep_param_vals) > 0:
            sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max() 
        else:
            sweep_min, sweep_max = 0, 1

        if plot:
            fig, axs = self._configure_subplot_mosaic(
                [['phase_raw'], ['phase_cal']], sweep_param_vals, width_ratios=[0.95, 0.05],
                sweep_label=sweep_label, sweep_cmap=sweep_cmap,
            )
            axs['phase_raw'].set(xlabel='Frequency (GHz)', ylabel=self.phase_ylabel)
            axs['phase_cal'].set(xlabel='Frequency (GHz)', ylabel=self.phase_ylabel)
            ret = fig, axs
        else:
            ret = None

        try: 
            target_indices = [(rg_arr[start_inds[i]], rgi_arr[start_inds[i]]) for i in inds]
            meta_keys = [k for k in keys if k.startswith(in_group + '/') and not k.endswith('/data')]
            for m_key in meta_keys:
                m_df = self.select(m_key)
                subset_df = m_df.loc[m_df.index.isin(target_indices)]
                sub_name = m_key.split('/')[-1]
                self.put(f"{tmp_out_prefix}/{sub_name}", subset_df, format='table')

            for i, val in zip(inds, sweep_param_vals):
                ind = start_inds[i]
                rg_val, rgi_val = rg_arr[ind], rgi_arr[ind] 
                
                data = self._get_group_values(data_group, i)
                
                if point_masks is not None and i in point_masks:
                    valid_rows = point_masks[i]
                    if 'RecordRow' in data.index.names:
                        mask = data.index.get_level_values('RecordRow').isin(valid_rows)
                        data = data.loc[mask]
                        
                if data.empty:
                    continue
                    
                I, Q, freqs = data.I.values, data.Q.values, data.frequency.values
                mlin = np.sqrt(I**2 + Q**2) 
                phase = np.unwrap(np.arctan2(Q, I)) 
                
                if lower_frequency_bound is not None:
                    lower_inds = np.where((lower_frequency_bound[0] <= freqs) * (freqs <= lower_frequency_bound[1]))[0] 
                    lower_freqs = freqs[lower_inds] 
                    lower_phase = phase[lower_inds]
                if upper_frequency_bound is not None:
                    upper_inds = np.where((upper_frequency_bound[0] <= freqs) * (freqs <= upper_frequency_bound[1]))[0] 
                    upper_freqs = freqs[upper_inds] 
                    upper_phase = phase[upper_inds] 
                    
                if fixed_coeffs is None: 
                    if (upper_frequency_bound is None) and (lower_frequency_bound is not None):
                        fit = Polynomial.fit(lower_freqs, lower_phase, degree)
                        fit = (lower_freqs, fit) 
                    elif (upper_frequency_bound is not None) and (lower_frequency_bound is None):
                        fit = Polynomial.fit(upper_freqs, upper_phase, degree)
                        fit = (upper_freqs, fit) 
                    elif (upper_frequency_bound is not None) and (lower_frequency_bound is not None):
                        fit = Polynomial.fit(np.concatenate([lower_freqs, upper_freqs]), np.concatenate([lower_phase, upper_phase]), degree) 
                        fit = (freqs, fit) 
                    else:
                        fit = (freqs, Polynomial.fit(freqs, phase, degree))
                elif domain is not None:
                    fit = Polynomial(fixed_coeffs, domain=domain) 
                    if (upper_frequency_bound is None) and (lower_frequency_bound is not None):
                        fit = (lower_freqs, fit) 
                    elif (upper_frequency_bound is not None) and (lower_frequency_bound is None):
                        fit = (upper_freqs, fit) 
                    else:
                        fit = (freqs, fit) 
                else:
                    raise ValueError('domain must be provided for a fixed polynomial fit.')

                fit_phase = np.zeros_like(freqs)
                fit_phase[(fit[0].min() <= freqs) * (freqs <= fit[0].max())] = fit[1](fit[0])
                cal_phase = phase - fit_phase
                cal_I, cal_Q = mlin*np.cos(cal_phase), mlin*np.sin(cal_phase)

                cal_df = pd.DataFrame({'frequency': freqs, 'I': cal_I, 'Q': cal_Q},
                    index=pd.MultiIndex.from_product(
                    [[rg_val], [rgi_val], ['%06i' % j for j in np.arange(freqs.shape[0])]],
                    names=['RecordGroup', 'RecordGroupInd', 'RecordRow'] 
                ))
                self.append(f"{tmp_out_prefix}/data", cal_df)
                
                cal_params_dict = {f'x{j}': fit[1].coef[j] for j in range(fit[1].coef.shape[0])} 
                cal_params_dict['domain_min'] = fit[1].domain.min()
                cal_params_dict['domain_max'] = fit[1].domain.max() 
                cal_params_df = pd.DataFrame(cal_params_dict, index=pd.MultiIndex.from_product(
                    [[rg_val], [rgi_val]], names=['RecordGroup', 'RecordGroupInd'] 
                ))
                self.append(f"{tmp_out_prefix}/polyphase_params", cal_params_df)

                if plot:
                    color = self._compute_color(val, sweep_min, sweep_max, sweep_cmap) 
                    axs['phase_raw'].plot(freqs, phase, color=color)
                    axs['phase_raw'].plot(freqs, fit_phase, ls=':', color='black') 
                    axs['phase_cal'].plot(freqs, cal_phase, color=color)

            out_keys = [k for k in self.keys() if k.startswith(out_group + '/')]
            for k in out_keys: self.remove(k)
                
            tmp_keys = [k for k in self.keys() if k.startswith(tmp_out_prefix)]
            for k in tmp_keys:
                sub_name = k.split('/')[-1]
                df = self.select(k)
                self.append(f"{out_group}/{sub_name}", df)
                self.remove(k)
                
            self._get_index_arrays(f"{out_group}/data")

        except Exception as e:
            for k in self.keys():
                if k.startswith(tmp_out_prefix): self.remove(k)
            raise e
        return ret

    def calibrate_from_file(self, filepath, in_group, out_group,
            bg_group='/base/data', inds=None, query=None, plot=False,
            sweep_param=None, sweep_cmap='viridis', sweep_label=None, 
        ):
        """
        Calibrate data using a trace from a separate background reference file.
        
        :param filepath: Path to the HDF5 file containing the background reference data.
        :type filepath: str
        :param in_group: The group path from which to pull data.
        :type in_group: str
        :param out_group: The output group path to write calibrated data into.
        :type out_group: str
        :param bg_group: The group path in the reference file pointing to the background trace.
        :type bg_group: str, optional
        :param inds: Indices over which to perform the calibration.
        :type inds: list or numpy.ndarray, optional
        :param query: Optional pandas query string to filter traces and points.
        :type query: str, optional
        :param plot: Generate a plot of the calibration.
        :type plot: bool, optional
        :param sweep_param: String to indicate a parameter that is swept over.
        :type sweep_param: str, optional
        :param sweep_cmap: Colormap to use.
        :type sweep_cmap: str, optional
        :param sweep_label: String label used to label the colorbar.
        :type sweep_label: str, optional
        """
        in_group = '/' + in_group.strip('/')
        out_group = '/' + out_group.strip('/')
        data_group = f"{in_group}/data"
        tmp_out_prefix = '/tmp_cal_process'
        
        bg_group = '/' + bg_group.strip('/')
        
        keys = self.keys()
        for k in keys:
            if k.startswith(tmp_out_prefix): self.remove(k)
            
        rg_arr, rgi_arr, rr_arr, start_inds = self._get_index_arrays(data_group)

        # - open background calibration data - # 
        with pd.HDFStore(filepath, mode='r') as bg_store:
            bg_data = bg_store.select(bg_group)
            
        bg_I, bg_Q, bg_freqs = bg_data.I.values, bg_data.Q.values, bg_data.frequency.values 
            
        bg_mlin = np.sqrt(bg_I**2 + bg_Q**2) 
        bg_mlog = (1 + 1*self.power)*10*np.log10(bg_mlin) 
        bg_phase = np.unwrap(np.arctan2(bg_Q, bg_I))

        if inds is None:
            inds = np.arange(start_inds.shape[0])
            
        point_masks = None
        if query is not None:
            inds, point_masks = self.group_query(in_group, query, inds)
            
        if sweep_param is None:
            sweep_param_vals = np.arange(start_inds.shape[0])[inds] if len(inds) > 0 else np.array([])
            param = 'iter' 
        else:
            subgroup, param = sweep_param.split('.') 
            sweep_path = f"{in_group}/{subgroup}"
            sweep_param_vals = self[sweep_path][param].values[inds]
            
        if len(sweep_param_vals) > 0:
            sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max() 
        else:
            sweep_min, sweep_max = 0, 1

        if plot:
            fig, axs = self._configure_subplot_mosaic(
                [['raw_mag', 'cal_mag'], ['raw_phase', 'cal_phase']],
                width_ratios=[0.475, 0.475, 0.05], sweep_cmap=sweep_cmap,
                sweep_label=sweep_label, sweep_param_vals=sweep_param_vals,
            )
            for key, ax in axs.items():
                ax.set_xlabel('Frequency (GHz)')
                if 'mag' in key: ax.set_ylabel(self.mag_ylabel)
                else: ax.set_ylabel(self.phase_ylabel)
            ret = fig, axs
        else:
            ret = None

        try: 
            target_indices = [(rg_arr[start_inds[i]], rgi_arr[start_inds[i]]) for i in inds]
            meta_keys = [k for k in keys if k.startswith(in_group + '/') and not k.endswith('/data')]
            for m_key in meta_keys:
                m_df = self.select(m_key)
                subset_df = m_df.loc[m_df.index.isin(target_indices)]
                sub_name = m_key.split('/')[-1]
                self.put(f"{tmp_out_prefix}/{sub_name}", subset_df, format='table')

            background_plotted = False 
            for i, val in zip(inds, sweep_param_vals):
                ind = start_inds[i]
                rg_val, rgi_val = rg_arr[ind], rgi_arr[ind]
                
                data = self._get_group_values(data_group, i)
                
                if point_masks is not None and i in point_masks:
                    valid_rows = point_masks[i]
                    if 'RecordRow' in data.index.names:
                        mask = data.index.get_level_values('RecordRow').isin(valid_rows)
                        data = data.loc[mask]
                        
                if data.empty:
                    continue
                    
                I, Q, freqs = data.I.values, data.Q.values, data.frequency.values
                mlin = np.sqrt(I**2 + Q**2)
                mlog = (1 + 1*self.power)*10*np.log10(mlin) 
                phase = np.unwrap(np.arctan2(Q, I))
                sdata = mlin*np.exp(1j*phase)
                
                bg_mlin_interp = np.interp(freqs, bg_freqs, bg_mlin) 
                bg_mlog_interp = (1 + 1*self.power)*10*np.log10(bg_mlin_interp) 
                bg_phase_interp = np.interp(freqs, bg_freqs, bg_phase) 
                bg_sdata_interp = bg_mlin_interp*np.exp(1j*bg_phase_interp) 

                cal_sdata = sdata / bg_sdata_interp 
                cal_I, cal_Q = np.real(cal_sdata), np.imag(cal_sdata)
                cal_mlog = (1 + 1*self.power)*10*np.log10(np.sqrt(cal_I**2 + cal_Q**2))  
                cal_phase = np.unwrap(np.arctan2(cal_Q, cal_I)) 
                
                cal_df = pd.DataFrame({'frequency': freqs, 'I': cal_I, 'Q': cal_Q}, 
                    index=pd.MultiIndex.from_product(
                    [[rg_val], [rgi_val], ['%06i' % j for j in np.arange(freqs.shape[0])]],
                    names=['RecordGroup', 'RecordGroupInd', 'RecordRow'] 
                ))
                self.append(f"{tmp_out_prefix}/data", cal_df) 

                if plot:
                    plot_freqs = freqs*1e-9 
                    color = self._compute_color(val, sweep_min, sweep_max, sweep_cmap) 
                    if not background_plotted:
                        axs['raw_mag'].plot(plot_freqs, bg_mlog_interp, ls=':', color='black')
                        axs['raw_phase'].plot(plot_freqs, bg_phase_interp, ls=':', color='black')
                        background_plotted = True
                    axs['raw_mag'].plot(plot_freqs, mlog, color=color) 
                    axs['raw_phase'].plot(plot_freqs, phase, color=color) 
                    axs['cal_mag'].plot(plot_freqs, cal_mlog, color=color)
                    axs['cal_phase'].plot(plot_freqs, cal_phase, color=color)

            out_keys = [k for k in self.keys() if k.startswith(out_group + '/')]
            for k in out_keys: self.remove(k)
                
            tmp_keys = [k for k in self.keys() if k.startswith(tmp_out_prefix)]
            for k in tmp_keys:
                sub_name = k.split('/')[-1]
                df = self.select(k)
                self.append(f"{out_group}/{sub_name}", df)
                self.remove(k)
                
            self._get_index_arrays(f"{out_group}/data")

        except Exception as e:
            for k in self.keys():
                if k.startswith(tmp_out_prefix): self.remove(k)
            raise e
        return ret

    # - RESONATOR PARAMETER FITTING -------------------------------------------------------------- #
    def fit_res_params(self,
            in_group, inds=None, query=None, plot=False, plot_text=False, 
            phase_fit_kwargs=None, fixed_Qc=None, sweep_param=None, 
            sweep_cmap='viridis', sweep_label=None, 
        ):
        """ 
        Fit resonator parameters and standard errors. Writes purely to `in_group/res_params`. 
        
        :param in_group: The group path from which to pull data (e.g., 'base').
        :type in_group: str
        :param inds: Record start indices to process over. If None, processes all available data.
        :type inds: list or numpy.ndarray, optional
        :param query: Optional pandas query string to filter traces and points.
        :type query: str, optional
        :param plot: Generate a plot of the fits.
        :type plot: bool, optional
        :param plot_text: Overlay the fit parameters as text on the plot.
        :type plot_text: bool, optional
        :param phase_fit_kwargs: Keyword arguments for the centered phase fit.
        :type phase_fit_kwargs: dict, optional
        :param fixed_Qc: Optionally fix the coupling quality factor.
        :type fixed_Qc: float, optional
        :param sweep_param: String to indicate a parameter that is swept over.
        :type sweep_param: str, optional
        :param sweep_cmap: Colormap to use.
        :type sweep_cmap: str, optional
        :param sweep_label: String label used to label the colorbar.
        :type sweep_label: str, optional
        """
        in_group = '/' + in_group.strip('/')
        data_group = f"{in_group}/data"
        
        rg_arr, rgi_arr, rr_arr, start_inds = self._get_index_arrays(data_group)

        if inds is None:
            inds = np.arange(start_inds.shape[0])
            
        point_masks = None
        if query is not None:
            inds, point_masks = self.group_query(in_group, query, inds)
            
        if sweep_param is None:
            sweep_param_vals = np.arange(start_inds.shape[0])[inds] if len(inds) > 0 else np.array([])
            param = 'iter' 
        else:
            subgroup, param = sweep_param.split('.') 
            sweep_path = f"{in_group}/{subgroup}"
            sweep_param_vals = self[sweep_path][param].values[inds]
            
        if len(sweep_param_vals) > 0:
            sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max() 
        else:
            sweep_min, sweep_max = 0, 1

        if plot:
            if plot_text:
                mosaic = [['iq', 'params'], ['centered_phase', 'centered_phase']]
            else:
                mosaic = [['iq'], ['centered_phase']] 
            fig, axs = self._configure_subplot_mosaic(
                mosaic, sweep_param_vals=sweep_param_vals, width_ratios=[0.9, 0.1],
                sweep_label=sweep_label, sweep_cmap=sweep_cmap,
            )
            axs['iq'].set(xlabel='I', ylabel='Q')
            axs['centered_phase'].set(xlabel='Frequency (GHz.)', ylabel=self.phase_ylabel)
            if plot_text:
                axs['params'].set_xticks([])
                axs['params'].set_yticks([])
                for key, spine in axs['params'].spines.items(): spine.set_visible(False)
            ret = fig, axs
        else:
            ret = None

        num_inds = len(inds)
        
        Ql_out = np.empty(num_inds, dtype=float)
        Qi_out = np.empty(num_inds, dtype=float)
        Qc_out = np.empty(num_inds, dtype=complex)
        phi_out = np.empty(num_inds, dtype=float)
        fr_out = np.empty(num_inds, dtype=float)
        
        Ql_err_out = np.empty(num_inds, dtype=float)
        Qi_err_out = np.empty(num_inds, dtype=float)
        Qc_err_out = np.empty(num_inds, dtype=float)
        fr_err_out = np.empty(num_inds, dtype=float)
        
        rg_out = np.empty(num_inds, dtype=object)
        rgi_out = np.empty(num_inds, dtype=object)
        
        valid_count = 0

        for i, val in zip(inds, sweep_param_vals):
            ind = start_inds[i]
            rg_val, rgi_val = rg_arr[ind], rgi_arr[ind]
            
            data = self._get_group_values(data_group, i)
            
            if point_masks is not None and i in point_masks:
                valid_rows = point_masks[i]
                if 'RecordRow' in data.index.names:
                    mask = data.index.get_level_values('RecordRow').isin(valid_rows)
                    data = data.loc[mask]
                    
            if data.empty:
                continue
                
            I, Q, freqs = data.I.values, data.Q.values, data.frequency.values
            mlin = np.sqrt(I**2 + Q**2) 
            phase = np.unwrap(np.arctan2(Q, I)) 
            sdata = mlin*np.exp(1j*phase)
            
            xc, yc, r = circle_fit(sdata)
            Icentered = I - xc
            Qcentered = Q - yc
            centered_phase = np.unwrap(np.arctan2(Qcentered, Icentered))
            
            phase_fit_kwargs = {} if phase_fit_kwargs is None else phase_fit_kwargs 
            
            try:
                params, pcov = self._centered_phase_fit(freqs, centered_phase, **phase_fit_kwargs)
                theta0, Ql, fr = params
                
                if np.any(np.isinf(pcov)) or np.any(np.isnan(pcov)):
                    raise ValueError("Invalid covariance matrix")
                    
                perr = np.sqrt(np.maximum(np.diag(pcov), 0))
                theta0_err, Ql_err, fr_err = perr
            except Exception:
                continue
            
            phi = -np.arcsin(yc/r)
            if self.geometry == 'hanger': 
                if fixed_Qc is None: 
                    Qc = Ql / (2*r*np.exp(-1j*phi))
                    Qcr = np.real(Qc)
                    Qi = 1 / ((1/Ql) - (1/Qcr))
                    
                    Qi_err = np.abs(Qi * (Ql_err / Ql)) if Ql != 0 else np.nan
                    Qc_err = np.abs(Qc * (Ql_err / Ql)) if Ql != 0 else np.nan
                else:
                    Qcr = fixed_Qc 
                    Qi = Qcr / (np.cos(phi) - 2*r)
                    Qc = Qcr + 1j*(Qi*Qcr*np.sin(phi) / (2*r*(Qi + Qcr)))
                    
                    Qi_err = 0.0
                    Qc_err = 0.0
            elif self.geometry == 'shunt':
                if fixed_Qc is None: 
                    Qc = 2*Ql / (2*r*np.exp(-1j*phi))
                    Qcr = np.real(Qc)
                    Qi = 1 / ((1/Ql) - (1/Qcr))
                    
                    Qi_err = np.abs(Qi * (Ql_err / Ql)) if Ql != 0 else np.nan
                    Qc_err = np.abs(Qc * (Ql_err / Ql)) if Ql != 0 else np.nan
                else:
                    Qcr = fixed_Qc 
                    Qi = Qcr / (np.cos(phi) - r)
                    Qc = Qcr + 1j*(Qi*Qcr*np.sin(phi) / (r*(Qi + Qcr)))
                    
                    Qi_err = 0.0
                    Qc_err = 0.0
                    
            Ql_out[valid_count] = Ql
            Qi_out[valid_count] = Qi
            Qc_out[valid_count] = Qc
            phi_out[valid_count] = phi
            fr_out[valid_count] = fr
            
            Ql_err_out[valid_count] = Ql_err
            Qi_err_out[valid_count] = Qi_err
            Qc_err_out[valid_count] = Qc_err
            fr_err_out[valid_count] = fr_err
            
            rg_out[valid_count] = rg_val
            rgi_out[valid_count] = rgi_val
            
            valid_count += 1

            if plot:
                plot_freqs = freqs * 1e-9
                phase_fit = self._centered_phase_func(freqs, theta0, Ql, fr) 
                color = self._compute_color(val, sweep_min, sweep_max, sweep_cmap)
                axs['iq'].scatter(I, Q, marker='.', color=color)
                circle = plt.Circle((xc, yc), r, edgecolor='r', facecolor='none', linewidth=2)
                axs['iq'].add_patch(circle)
                axs['centered_phase'].plot(plot_freqs, centered_phase, color=color)
                axs['centered_phase'].plot(plot_freqs, phase_fit, ls=':', color='black')
                if plot_text:
                    params_str = '\n'.join([
                        r'$Q_l = %0.2f \pm %0.2f$' % (Ql, Ql_err),
                        r'$Q_i = %0.2f \pm %0.2f$' % (Qi, Qi_err),
                        r'$Q_{cr} = %0.2f \pm %0.2f$' % (np.real(Qc), Qc_err),
                        r'$\phi = %0.2f$' % phi,
                        r'$f_r = %0.6f \pm %0.6f$ (GHz.)' % (fr*1e-9, fr_err*1e-9)
                    ])
                    axs['params'].text(0.1, 0.2, params_str, fontsize=14)

        if valid_count > 0:
            res_params_df = pd.DataFrame({
                'Ql': Ql_out[:valid_count], 
                'Qi': Qi_out[:valid_count], 
                'Qc': Qc_out[:valid_count], 
                'phi': phi_out[:valid_count], 
                'fr': fr_out[:valid_count],
                'Ql_err': Ql_err_out[:valid_count],
                'Qi_err': Qi_err_out[:valid_count],
                'Qc_err': Qc_err_out[:valid_count],
                'fr_err': fr_err_out[:valid_count]
            })
            
            res_params_df.index = pd.MultiIndex.from_arrays(
                [rg_out[:valid_count], rgi_out[:valid_count]], 
                names=['RecordGroup', 'RecordGroupInd']
            )
            
            target_key = f"{in_group}/res_params"
            if target_key in self.keys():
                self.remove(target_key)
                
            self.append(target_key, res_params_df)

        return ret

    def find_min_mag_fr(self, in_group, inds=None, query=None):
            """ 
            Find the resonance frequency by identifying the minimum magnitude of the I/Q data.
            Writes the results to `in_group/min_mag_params`.
            
            :param in_group: The input group path from which to pull data (e.g., 'base').
            :type in_group: str
            :param inds: Record start indices to process over. If None, processes all available data.
            :type inds: list or numpy.ndarray, optional
            :param query: Optional pandas query string to filter traces and points.
            :type query: str, optional
            """
            in_group = '/' + in_group.strip('/')
            data_group = f"{in_group}/data"
            
            rg_arr, rgi_arr, rr_arr, start_inds = self._get_index_arrays(data_group)

            if inds is None:
                inds = np.arange(start_inds.shape[0])
                
            point_masks = None
            if query is not None:
                inds, point_masks = self.group_query(in_group, query, inds)

            num_inds = len(inds)
            
            fr_out = np.empty(num_inds, dtype=float)
            rg_out = np.empty(num_inds, dtype=object)
            rgi_out = np.empty(num_inds, dtype=object)
            
            valid_count = 0

            for i in inds:
                ind = start_inds[i]
                rg_val, rgi_val = rg_arr[ind], rgi_arr[ind]
                
                data = self._get_group_values(data_group, i)
                
                if point_masks is not None and i in point_masks:
                    valid_rows = point_masks[i]
                    if 'RecordRow' in data.index.names:
                        mask = data.index.get_level_values('RecordRow').isin(valid_rows)
                        data = data.loc[mask]
                        
                if data.empty:
                    continue
                    
                I, Q, freqs = data.I.values, data.Q.values, data.frequency.values
                
                mlin = np.sqrt(I**2 + Q**2)
                
                min_idx = np.argmin(mlin)
                
                fr_out[valid_count] = freqs[min_idx]
                rg_out[valid_count] = rg_val
                rgi_out[valid_count] = rgi_val
                
                valid_count += 1

            if valid_count > 0:
                fr_out = fr_out[:valid_count]
                rg_out = rg_out[:valid_count]
                rgi_out = rgi_out[:valid_count]
                
                min_mag_df = pd.DataFrame({'fr': fr_out})
                
                min_mag_df.index = pd.MultiIndex.from_arrays(
                    [rg_out, rgi_out], names=['RecordGroup', 'RecordGroupInd']
                )
                
                out_path = f"{in_group}/min_mag_params"
                
                if out_path in self.keys():
                    self.remove(out_path)
                    
                self.append(out_path, min_mag_df)

    def find_max_mag_bw(self, in_group='base', inds=None, query=None):
        """
        Find the resonance frequency (peak), bandwidth (FWHM), and Quality Factor (Q) 
        for transmission data. The bandwidth is calculated at the half-power level 
        (max_magnitude / sqrt(2)) using linear interpolation.
        
        Writes the results to `in_group/max_mag_params`.

        :param in_group: The input group path from which to pull data (e.g., 'base').
        :type in_group: str
        :param inds: Record start indices to process over. If None, processes all available data.
        :type inds: list or numpy.ndarray, optional
        :param query: Optional pandas query string to filter traces and points.
        :type query: str, optional
        """
        in_group = '/' + in_group.strip('/')
        data_group = f"{in_group}/data"
        
        # Get structural index arrays
        rg_arr, rgi_arr, rr_arr, start_inds = self._get_index_arrays(data_group)

        if inds is None:
            inds = np.arange(start_inds.shape[0])
            
        point_masks = None
        if query is not None:
            inds, point_masks = self.group_query(in_group, query, inds)

        num_inds = len(inds)
        
        # Pre-allocate arrays for performance
        max_out = np.empty(num_inds, dtype=float)
        f_max_out = np.empty(num_inds, dtype=float)
        bw_out = np.empty(num_inds, dtype=float)
        q_out = np.empty(num_inds, dtype=float)
        rg_out = np.empty(num_inds, dtype=object)
        rgi_out = np.empty(num_inds, dtype=object)
        
        valid_count = 0

        for i in inds:
            ind = start_inds[i]
            rg_val, rgi_val = rg_arr[ind], rgi_arr[ind]
            
            data = self._get_group_values(data_group, i)
            
            if point_masks is not None and i in point_masks:
                valid_rows = point_masks[i]
                if 'RecordRow' in data.index.names:
                    mask = data.index.get_level_values('RecordRow').isin(valid_rows)
                    data = data.loc[mask]
                    
            if data.empty or len(data) < 3:
                continue
                
            I, Q, freqs = data.I.values, data.Q.values, data.frequency.values
            mag = np.sqrt(I**2 + Q**2)
            
            # 1. Find peak frequency
            max_mag = mag.max()
            max_idx = np.argmax(mag)
            m_max = mag[max_idx]
            f_max = freqs[max_idx]
            
            # 2. Define half-power target (magnitude / sqrt(2))
            target = m_max / np.sqrt(2)
            
            # 3. Find bandwidth via interpolation
            mag_left = mag[:max_idx+1]
            freq_left = freqs[:max_idx+1]
            mag_right = mag[max_idx:]
            freq_right = freqs[max_idx:]
            
            try:
                # Interpolate to find f_low and f_high
                f_low = np.interp(target, mag_left, freq_left)
                f_high = np.interp(target, mag_right[::-1], freq_right[::-1])
                
                bw = f_high - f_low
                q_val = f_max / bw if bw != 0 else np.nan
            except Exception:
                bw, q_val = np.nan, np.nan

            # Store results in pre-allocated arrays
            max_out[valid_count] = max_mag 
            f_max_out[valid_count] = f_max
            bw_out[valid_count] = bw
            q_out[valid_count] = q_val
            rg_out[valid_count] = rg_val
            rgi_out[valid_count] = rgi_val
            
            valid_count += 1

        if valid_count > 0:
            # Slice arrays to the actual number of valid records processed
            max_out = max_out[:valid_count] 
            f_max_out = f_max_out[:valid_count]
            bw_out = bw_out[:valid_count]
            q_out = q_out[:valid_count]
            rg_out = rg_out[:valid_count]
            rgi_out = rgi_out[:valid_count]
            
            # Create result DataFrame
            max_mag_df = pd.DataFrame({
                'max_mag': max_out, 
                'f_max': f_max_out,
                'bw': bw_out,
                'Q': q_out
            })
            
            max_mag_df.index = pd.MultiIndex.from_arrays(
                [rg_out, rgi_out], names=['RecordGroup', 'RecordGroupInd']
            )
            
            out_path = f"{in_group}/max_mag_params"
            
            # Parameter-only output: Replace existing results in the same group
            if out_path in self.keys():
                self.remove(out_path)
                
            self.append(out_path, max_mag_df)
