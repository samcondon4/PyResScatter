import os
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
        self.mag_ylabel = r'$|S_{%s}|$' % self.sparam 
        self.phase_ylabel = r'$\angle S_{%s}$ (rads)' % self.sparam 
        
        self._index_cache = {}
        
        # - Clean up failed temporary groups from previous run crashes - #
        keys = self.keys()
        for k in keys:
            if k.startswith('/tmp_') or '/tmp_avg_process' in k:
                self.remove(k)
        # (Note: Removing tmp files here will leave "dead space", but you can always
        # run ResonatorScatteringStore.repack('my_file.h5') manually later if it gets bloated).

        # - Cache indices for the base data group - #
        if '/base/data' in self.keys() or 'base/data' in self.keys():
            self._get_index_arrays('/base/data')

    # - STATIC METHODS ------------------------------------------------------------------------------------------- #
    @staticmethod
    def _repack(filepath, key_mapping=None):
        """ Repack an HDF5 file to reclaim disk space.
        
        :param filepath: Path to the HDF5 file.
        :param key_mapping: Optional dictionary mapping old keys to new keys to rename groups 
                            during the repacking process without doubling file size.
        """
        if not os.path.exists(filepath):
            return
            
        key_mapping = key_mapping or {}
        
        # Create tmp file in the same directory to ensure atomic os.replace across filesystems
        dir_name = os.path.dirname(os.path.abspath(filepath))
        tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.h5')
        os.close(tmp_fd) # Close OS-level file descriptor so pandas can open it safely
        
        try:
            with pd.HDFStore(filepath, mode='r') as store_in, pd.HDFStore(tmp_path, mode='w') as store_out:
                for key in store_in.keys():
                    df = store_in.select(key)
                    new_key = key_mapping.get(key, key)
                    store_out.put(new_key, df, format='table')
                    
            # Atomically replace the bloated file with the fresh, repacked file
            os.replace(tmp_path, filepath)
        except Exception as e:
            # Clean up the temp file if something fails
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise e
    
    @staticmethod
    def _compute_color(val, vmin, vmax, cmap='viridis'):
        """ Convert a value between a minimum and maximum to an integer between
        0 and 256 for use in a colormap.
        
        :param val: Integer value to convert to a color.
        :param vmin: Minimum integer value that val can take.
        :param vmax: Maximum integer value that val can take.
        :param cmap: String identifying the colormap to use.
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
        """ Configure a subplot mosaic and colorbar.

        :param mosaic: List used as input to the subplot_mosaic call.
        :param sweep_param_vals: Array with the parameter sweep data.
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
    def _get_group_values(self, group, index, param=None, frequency_bound=None, index_group=None):
            """ Return the dataframe from the group at the specified index.

            :param group: String corresponding to the hierarchical HDF group to pull the dataframe from.
            :param index: Index from the RecordGroup and RecordGroupInd list to pull dataframe from.
            :param param: Parameter to pull from the dataframe. 
            :param frequency_bound: Frequency limits to take HDF group data between. 
            :param index_group: The baseline data group whose indexing scheme should be used. 
                                If None, defaults to the 'data' group in the same parent directory.
            """
            # Ensure consistent absolute pathing
            group = '/' + group.strip('/')
            
            if index_group is None:
                # If no index_group provided, guess the corresponding data group. 
                # e.g., '/base/meta' -> '/base/data'
                parent_path = group.rsplit('/', 1)[0]
                index_group = f"{parent_path}/data"
            else:
                index_group = '/' + index_group.strip('/')
                
            rg_arr, rgi_arr, rr_arr, start_inds = self._get_index_arrays(index_group)
            ind = start_inds[index]
            
            try:
                iter(ind)
            except TypeError:
                rg, rgi = rg_arr[ind], rgi_arr[ind] 
                where_str = f'RecordGroup == "{rg}" & RecordGroupInd == "{rgi}"'
            else:
                rg, rgi = rg_arr[ind], rgi_arr[ind] 
                rgmin, rgmax = rg.min(), rg.max()
                rgimin, rgimax = rgi.min(), rgi.max()
                where_str = ' & '.join([
                        f'RecordGroup >= "{rgmin}" & RecordGroup <= "{rgmax}"',
                        f'RecordGroupInd >= "{rgimin}" & RecordGroupInd <= "{rgimax}"'
                    ])
                    
            df = self.select(group, where=where_str)
            
            if frequency_bound is not None and 'frequency' in df.columns:
                freqs = df.frequency.values 
                inds_bound = (frequency_bound[0] < freqs) * (freqs < frequency_bound[1])
                df = df.iloc[inds_bound]
                
            if param is not None:
                ret = df[param].values
            else:
                ret = df

            return ret
    
    def _get_index_arrays(self, data_group):
        """ Read and cache the index structures for any data group in the hierarchy. 
        
        :param data_group: String corresponding to the hierarchical path of the data group.
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
    def average_traces_on_sweep(self, in_group, out_group, sweep_param):
        """ Average traces based on unique values of a given sweep parameter.
        
        :param in_group: The input group path from which to pull data (e.g., 'base').
        :param out_group: The output group path to write the averaged traces into (e.g., 'process_0'). 
                          Supports overwriting if in_group == out_group.
        :param sweep_param: String formatted as 'subgroup.column' to map uniqueness over 
                            (e.g., 'meta.power').
        """
        # Ensure paths have a consistent absolute format (e.g. '/base')
        in_group = '/' + in_group.strip('/')
        out_group = '/' + out_group.strip('/')
        
        subgroup, param = sweep_param.split('.')
        sweep_path = f"{in_group}/{subgroup}"
        
        # Load the sweep parameter dataframe
        sweep_df = self.select(sweep_path)
        unique_vals = pd.unique(sweep_df[param])
        
        # Identify and load all metadata-like groups (everything under in_group except data)
        # This dynamically captures /meta, /proc_params, or anything else alongside /data
        keys = self.keys()
        meta_keys = [k for k in keys if k.startswith(in_group + '/') and not k.endswith('/data')]
        meta_dfs = {k: self.select(k) for k in meta_keys}
        
        data_group = f"{in_group}/data"
        tmp_out_prefix = '/tmp_avg_process'
        
        # Clear any leftover tmp groups from a previously crashed run
        for k in keys:
            if k.startswith(tmp_out_prefix):
                self.remove(k)
                
        for i, val in enumerate(unique_vals):
            matching_sweep_df = sweep_df[sweep_df[param] == val]
            
            I_list, Q_list = [], []
            freqs = None
            
            # Retrieve traces for this sweep value
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
            
            # Calculate standard error of the mean (SEM)
            if N > 1:
                I_err = np.std(I_list, axis=0, ddof=1) / np.sqrt(N)
                Q_err = np.std(Q_list, axis=0, ddof=1) / np.sqrt(N)
            else:
                # If only one trace exists, standard error is zero
                I_err = np.zeros_like(I_avg)
                Q_err = np.zeros_like(Q_avg)
                
            new_rg, new_rgi = '000000', '%06i' % i
            
            # Write Averaged Data to temporary group
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
            
            # Write Processed Metadata and Proc Params to temporary groups
            meta_index = pd.MultiIndex.from_product(
                [[new_rg], [new_rgi]],
                names=['RecordGroup', 'RecordGroupInd']
            )
            
            first_idx = matching_sweep_df.index[0]
            
            for m_key, m_df in meta_dfs.items():
                if first_idx in m_df.index:
                    new_row = m_df.loc[[first_idx]].copy()
                    new_row.index = meta_index
                    
                    # Ensure the sweep param explicitly reflects this value
                    if m_key == sweep_path:
                        new_row[param] = val
                        
                    sub_name = m_key.split('/')[-1]
                    self.append(f"{tmp_out_prefix}/{sub_name}", new_row)
                    
        # Replace out_group with the newly generated temporary groups
        # We delete the destination keys first to cleanly support 'in_group == out_group' overwriting
        out_keys = [k for k in self.keys() if k.startswith(out_group + '/')]
        for k in out_keys:
            self.remove(k)
            
        # Move temporary groups into the final out_group
        tmp_keys = [k for k in self.keys() if k.startswith(tmp_out_prefix)]
        for k in tmp_keys:
            sub_name = k.split('/')[-1]
            df = self.select(k)
            self.append(f"{out_group}/{sub_name}", df)
            self.remove(k)
            
        # Cache the index for the newly generated output data group
        self._get_index_arrays(f"{out_group}/data")

    def average_res_params_on_sweep(self, in_group, sweep_param):
        """ Average the fitted resonator parameters based on unique values of a given sweep parameter.
        
        :param in_group: The input group path from which to pull data (e.g., 'base').
        :param sweep_param: String formatted as 'subgroup.column' to map uniqueness over 
                            (e.g., 'meta.power').
        """
        in_group = '/' + in_group.strip('/')
        params_group = f"{in_group}/res_params"
        
        if params_group not in self.keys():
            raise KeyError(f"No res_params found at {params_group}. Please run fit_res_params first.")
            
        subgroup, param = sweep_param.split('.')
        sweep_path = f"{in_group}/{subgroup}"
        
        sweep_df = self.select(sweep_path)
        res_df = self.select(params_group)
        
        unique_vals = pd.unique(sweep_df[param])
        
        avg_records = []
        index_tuples = []
        
        for val in unique_vals:
            # Find matching indices in the sweep metadata
            matching_indices = sweep_df[sweep_df[param] == val].index
            
            # Intersect with the indices that actually have fitted parameters
            # (In case some trace fits failed and aren't in res_df)
            valid_indices = matching_indices.intersection(res_df.index)
            
            if len(valid_indices) == 0:
                continue
                
            subset_df = res_df.loc[valid_indices]
            N = len(subset_df)
            
            row = {}
            # - Average resonator parameters - #
            for col in subset_df.columns:
                vals = subset_df[col].values
                row[col] = np.mean(vals)
                
                # Calculate standard error of the mean (SEM)
                if N > 1:
                    row[f"{col}_err"] = np.std(vals, ddof=1) / np.sqrt(N)
                else:
                    row[f"{col}_err"] = 0.0
            
            # - Average the sweep parameter itself - #
            sweep_vals = sweep_df.loc[valid_indices, param].values
            row[param] = np.mean(sweep_vals)
            
            if N > 1:
                row[f"{param}_err"] = np.std(sweep_vals, ddof=1) / np.sqrt(N)
            else:
                row[f"{param}_err"] = 0.0
                    
            avg_records.append(row)
            
            # Inherit the exact MultiIndex (RecordGroup, RecordGroupInd) of the FIRST trace in the subselection
            index_tuples.append(valid_indices[0])
            
        if avg_records:
            avg_res_df = pd.DataFrame(avg_records)
            avg_res_df.index = pd.MultiIndex.from_tuples(index_tuples, names=['RecordGroup', 'RecordGroupInd'])
            
            # Safely overwrite if it already exists
            out_path = f"{in_group}/avg_res_params"
            if out_path in self.keys():
                self.remove(out_path)
                
            self.append(out_path, avg_res_df)

    # - CALIBRATION FUNCTIONS -------------------------------------------------------------------------- #
    def calibrate_cable_delay(self, 
                in_group, out_group,
                tau=None, offset=None, 
                frequency_bound=None, fit_frequency_bound=None, inds=None, 
                plot=False, sweep_param=None, sweep_cmap='viridis', sweep_label=None,
            ):
            """ Remove a line from the unwrapped phase data.
            
            :param in_group: The group path from which to pull data (e.g., 'base').
            :param out_group: The output group path to write the calibrated data into (e.g., 'cal_cable').
                            Supports overwriting if in_group == out_group.
            :param tau: Fixed cable delay slope. If None a line will be fit to the unwrapped phase.
            :param offset: Fixed cable delay offset.
            :param frequency_bound: Frequency range over which calibration should be performed. 
            :param fit_frequency_bound: Frequency range over which a line fit should be performed. 
            :param inds: Indices over which to perform the calibration. 
            :param plot: Boolean to indicate if a plot showing the calibration results should be generated. 
            :param sweep_param: String to indicate a parameter that is swept over in the data.
            :param sweep_cmap: Colormap to use to indicate the value of the swept parameter.
            :param sweep_label: String label used to label the colorbar. 
            """
            # - Format paths - #
            in_group = '/' + in_group.strip('/')
            out_group = '/' + out_group.strip('/')
            data_group = f"{in_group}/data"
            tmp_out_prefix = '/tmp_cal_process'
            
            # - Clean any leftover temp groups from a previously crashed run - #
            keys = self.keys()
            for k in keys:
                if k.startswith(tmp_out_prefix):
                    self.remove(k)

            rg_arr, rgi_arr, rr_arr, start_inds = self._get_index_arrays(data_group)

            # - apply indices ------------------------------------------------------------- # 
            if inds is None:
                inds = np.arange(start_inds.shape[0])

            # - set up sweep parameter and plot ------------------------------------------- #
            if sweep_param is None:
                sweep_param_vals = np.arange(start_inds.shape[0])
                param = 'iter' 
            else:
                subgroup, param = sweep_param.split('.') 
                sweep_path = f"{in_group}/{subgroup}"
                sweep_param_vals = self[sweep_path][param].values[inds]
                
            sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max()
            
            if plot:
                fig, axs = self._configure_subplot_mosaic(
                    [['phase_raw', 'iq_raw'], ['phase_cal', 'iq_cal']],
                    sweep_param_vals,
                    width_ratios=[0.475, 0.475, 0.05],
                    sweep_label=sweep_label,
                    sweep_cmap=sweep_cmap,
                )
                axs['phase_cal'].set(xlabel='Frequency (GHz.)', ylabel=self.phase_ylabel)
                axs['phase_raw'].set(xlabel='Frequency (GHz.)', ylabel=self.phase_ylabel)
                axs['iq_cal'].set(ylabel='Q', xlabel='I')
                axs['iq_raw'].set(ylabel='Q', xlabel='I')
                ret = fig, axs
            else:
                ret = None

            try: 
                # - Copy over non-data metadata (meta, proc_params, etc.) in bulk for speed - #
                target_indices = [(rg_arr[start_inds[i]], rgi_arr[start_inds[i]]) for i in inds]
                meta_keys = [k for k in keys if k.startswith(in_group + '/') and not k.endswith('/data')]
                for m_key in meta_keys:
                    m_df = self.select(m_key)
                    # Filter to only the indices we are actually calibrating
                    subset_df = m_df.loc[m_df.index.isin(target_indices)]
                    sub_name = m_key.split('/')[-1]
                    self.put(f"{tmp_out_prefix}/{sub_name}", subset_df, format='table')
                
                # - Process and calibrate data iteratively ------------------------------------ # 
                for i, val in zip(inds, sweep_param_vals): 
                    ind = start_inds[i] 
                    rg_val, rgi_val = rg_arr[ind], rgi_arr[ind] 

                    # Fetch raw trace
                    data = self._get_group_values(data_group, i, frequency_bound=frequency_bound) 
                    freqs = data.frequency.values  
                    I, Q = data.I.values, data.Q.values
                    phase = np.unwrap(np.arctan2(Q, I))
                    mlin = np.sqrt(I**2 + Q**2)
                    
                    # Apply fitting bounds if provided
                    if fit_frequency_bound is not None:
                        fit_inds_arr = (fit_frequency_bound[0] < freqs) * (freqs < fit_frequency_bound[1])
                        fit_freqs = freqs[fit_inds_arr]
                        fit_phase = phase[fit_inds_arr]
                    else:
                        fit_freqs = freqs
                        fit_phase = phase
                        
                    # Fit cable delay parameters
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
                        
                    # Apply calibration
                    line = self._line_func(freqs, tau_fit, offset_fit)
                    corrected_phase = phase - line
                    Ical, Qcal = mlin*np.cos(corrected_phase), mlin*np.sin(corrected_phase)
                    
                    # Write calibrated data to temporary store
                    cal_df = pd.DataFrame(
                        {'frequency': freqs, 'I': Ical, 'Q': Qcal}, 
                        index=pd.MultiIndex.from_product(
                            [[rg_val], [rgi_val], ['%06i' % j for j in np.arange(freqs.shape[0])]],
                            names=['RecordGroup', 'RecordGroupInd', 'RecordRow'] 
                        )
                    )
                    self.append(f"{tmp_out_prefix}/data", cal_df) 
                    
                    # Write fitted parameters to temporary store
                    params_df = pd.DataFrame(
                        {'tau': tau_fit, 'cable_delay_offset': offset_fit}, 
                        index=pd.MultiIndex.from_product(
                            [[rg_val], [rgi_val]], names=['RecordGroup', 'RecordGroupInd']
                        )
                    ) 
                    self.append(f"{tmp_out_prefix}/cable_delay_params", params_df) 
                    
                    # - Plotting - #
                    if plot:
                        plot_freqs = freqs*1e-9
                        color = self._compute_color(val, sweep_min, sweep_max, sweep_cmap) 
                        axs['phase_raw'].plot(plot_freqs, phase, color=color)
                        axs['phase_raw'].plot(plot_freqs, line, ls=':', color='black')
                        axs['phase_cal'].plot(plot_freqs, corrected_phase, color=color)
                        axs['iq_raw'].scatter(I, Q, color=color, marker='.')
                        axs['iq_cal'].scatter(Ical, Qcal, color=color, marker='.')

                # - Move temporary groups into the final out_group ---------------------------- #
                # Remove destination keys if overwriting
                out_keys = [k for k in self.keys() if k.startswith(out_group + '/')]
                for k in out_keys:
                    self.remove(k)
                    
                # Rename temp groups
                tmp_keys = [k for k in self.keys() if k.startswith(tmp_out_prefix)]
                for k in tmp_keys:
                    sub_name = k.split('/')[-1]
                    df = self.select(k)
                    self.append(f"{out_group}/{sub_name}", df)
                    self.remove(k)
                    
                # Update index cache for the new data group
                self._get_index_arrays(f"{out_group}/data")
            
            except Exception as e:
                # - Clean up the temp data groups if the fit errored - # 
                for k in self.keys():
                    if k.startswith(tmp_out_prefix):
                        self.remove(k)
                raise e

            return ret
    
    def calibrate_constant_scaling(self,
                in_group, out_group,
                a=None, alpha=None, phase_fit_kwargs=None,
                inds=None, plot=False, frequency_bound=None,
                sweep_param=None, sweep_cmap='viridis', sweep_label=None, 
            ):
            # - Format paths - #
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
                
            if sweep_param is None:
                sweep_param_vals = np.arange(start_inds.shape[0])
                param = 'iter' 
            else:
                subgroup, param = sweep_param.split('.') 
                sweep_path = f"{in_group}/{subgroup}"
                sweep_param_vals = self[sweep_path][param].values[inds]
                
            sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max() 

            if plot:
                fig, axs = self._configure_subplot_mosaic(
                    [['iq_raw', 'iq_process'], ['centered_phase', 'iq_final']],
                    sweep_param_vals=sweep_param_vals, width_ratios=[0.475, 0.475, 0.05],
                    sweep_label=sweep_label, sweep_cmap=sweep_cmap,
                )
                for key, ax in axs.items():
                    if 'iq' in key:
                        ax.set(xlabel='I', ylabel='Q')
                axs['centered_phase'].set(xlabel='Frequency (GHz.)', ylabel=self.phase_ylabel) 
                ret = fig, axs
            else:
                ret = None

            try: 
                # - Bulk copy metadata - #
                target_indices = [(rg_arr[start_inds[i]], rgi_arr[start_inds[i]]) for i in inds]
                meta_keys = [k for k in keys if k.startswith(in_group + '/') and not k.endswith('/data')]
                for m_key in meta_keys:
                    m_df = self.select(m_key)
                    subset_df = m_df.loc[m_df.index.isin(target_indices)]
                    sub_name = m_key.split('/')[-1]
                    self.put(f"{tmp_out_prefix}/{sub_name}", subset_df, format='table')
                    
                j = 0 
                for i, val in zip(inds, sweep_param_vals):
                    ind = start_inds[i]
                    rg_val, rgi_val = rg_arr[ind], rgi_arr[ind]
                    fb = frequency_bound[j] if (frequency_bound is not None and len(frequency_bound) > 2) else frequency_bound
                    
                    data = self._get_group_values(data_group, i, frequency_bound=fb)
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
                    j += 1

                # - Move to final group - #
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
                frequency_bound=None, lower_frequency_bound=None, upper_frequency_bound=None, 
                inds=None, degree=2, fixed_coeffs=None, domain=None,
                plot=False, sweep_param=None, sweep_cmap='viridis', sweep_label=None, 
            ):
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
                
            if sweep_param is None:
                sweep_param_vals = np.arange(start_inds.shape[0])
                param = 'iter' 
            else:
                subgroup, param = sweep_param.split('.') 
                sweep_path = f"{in_group}/{subgroup}"
                sweep_param_vals = self[sweep_path][param].values[inds]
                
            sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max() 

            if plot:
                fig, axs = self._configure_subplot_mosaic(
                    [['mag_raw'], ['mag_cal']], sweep_param_vals, width_ratios=[0.95, 0.05],
                    sweep_label=sweep_label, sweep_cmap=sweep_cmap,
                )
                axs['mag_raw'].set(xlabel='Frequency (GHz.)', ylabel=self.mag_ylabel)
                axs['mag_cal'].set(xlabel='Frequency (GHz.)', ylabel=self.mag_ylabel)
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
                    
                    data = self._get_group_values(data_group, i, frequency_bound=frequency_bound)
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
            frequency_bound=None, lower_frequency_bound=None, upper_frequency_bound=None,
            inds=None, degree=2, fixed_coeffs=None, domain=None, plot=False, 
            sweep_param=None, sweep_cmap='viridis', sweep_label=None,
        ):
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
            
        if sweep_param is None:
            sweep_param_vals = np.arange(start_inds.shape[0])
            param = 'iter' 
        else:
            subgroup, param = sweep_param.split('.') 
            sweep_path = f"{in_group}/{subgroup}"
            sweep_param_vals = self[sweep_path][param].values[inds]
            
        sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max() 

        if plot:
            fig, axs = self._configure_subplot_mosaic(
                [['phase_raw'], ['phase_cal']], sweep_param_vals, width_ratios=[0.95, 0.05],
                sweep_label=sweep_label, sweep_cmap=sweep_cmap,
            )
            axs['phase_raw'].set(xlabel='Frequency (GHz.)', ylabel=self.phase_ylabel)
            axs['phase_cal'].set(xlabel='Frequency (GHz.)', ylabel=self.phase_ylabel)
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
                
                data = self._get_group_values(data_group, i, frequency_bound=frequency_bound)
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
            bg_group='/base/data', frequency_bound=None, inds=None, plot=False,
            sweep_param=None, sweep_cmap='viridis', sweep_label=None, 
        ):
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
        if frequency_bound is not None:
            bg_inds = np.where((frequency_bound[0] <= bg_freqs) * (bg_freqs <= frequency_bound[1]))[0] 
            bg_I, bg_Q, bg_freqs = bg_I[bg_inds], bg_Q[bg_inds], bg_freqs[bg_inds] 
            
        bg_mlin = np.sqrt(bg_I**2 + bg_Q**2) 
        bg_mlog = (1 + 1*self.power)*10*np.log10(bg_mlin) 
        bg_phase = np.unwrap(np.arctan2(bg_Q, bg_I))

        if inds is None:
            inds = np.arange(start_inds.shape[0])
            
        if sweep_param is None:
            sweep_param_vals = np.arange(start_inds.shape[0])
            param = 'iter' 
        else:
            subgroup, param = sweep_param.split('.') 
            sweep_path = f"{in_group}/{subgroup}"
            sweep_param_vals = self[sweep_path][param].values[inds]
            
        sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max() 

        if plot:
            fig, axs = self._configure_subplot_mosaic(
                [['raw_mag', 'cal_mag'], ['raw_phase', 'cal_phase']],
                width_ratios=[0.475, 0.475, 0.05], sweep_cmap=sweep_cmap,
                sweep_label=sweep_label, sweep_param_vals=sweep_param_vals,
            )
            for key, ax in axs.items():
                ax.set_xlabel('Frequency (GHz.)')
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
                
                data = self._get_group_values(data_group, i, frequency_bound=frequency_bound)
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

    # - PLOTTING FUNCTIONS ----------------------------------------------------------------------------- #
    def plot_mag_phase(self,
            in_group='base', frequency_bound=None, inds=None,
            sweep_param=None, sweep_cmap='viridis', sweep_label=None,
        ):
        """ Plot the magnitude and phase for a specified group in the hierarchy.
        
        :param in_group: The group path from which to pull data (e.g., 'base' or 'process_0').
        :param frequency_bound: Frequency range over which to plot.
        :param inds: Record start indices to plot over. If None, all available data will be plotted.
        :param sweep_param: String formatted as 'subgroup.column' to map uniqueness over.
        :param sweep_cmap: Colormap used to indicate the value of the swept parameter.
        :param sweep_label: String label used to indicate the colorbar.
        """
        # Format paths for the targeted hierarchy
        in_group = '/' + in_group.strip('/')
        data_group = f"{in_group}/data"
        
        rg_arr, rgi_arr, rr_arr, start_inds = self._get_index_arrays(data_group)

        # - apply indices --------------- #
        if inds is None:
            inds = np.arange(start_inds.shape[0])
            
        if sweep_param is None:
            sweep_param_vals = np.arange(start_inds.shape[0])
            param = 'iter' 
        else:
            subgroup, param = sweep_param.split('.') 
            sweep_group_path = f"{in_group}/{subgroup}"
            sweep_param_vals = self[sweep_group_path][param].values[inds]
            
        sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max() 

        # - configure figure and axes objects - #
        fig, axs = self._configure_subplot_mosaic(
            [['mag'], ['phase']],
            sweep_param_vals,
            width_ratios=[0.95, 0.05],
            sweep_label=sweep_label,
            sweep_cmap=sweep_cmap,
        )
            
        axs['mag'].set_xticks([]) 
        axs['phase'].set_xlabel('Frequency (GHz.)')
        axs['mag'].set_ylabel(self.mag_ylabel)
        axs['phase'].set_ylabel(self.phase_ylabel)

        # - plot - #
        for i, val in zip(inds, sweep_param_vals):
            # Fetch data directly from the target group
            data = self._get_group_values(data_group, i, frequency_bound=frequency_bound)
            I, Q, freqs = data.I.values, data.Q.values, data.frequency.values 
            freqs *= 1e-9 
            mlog = (1 + self.power*1)*10*np.log10(np.sqrt(I**2 + Q**2)) 
            phase = np.unwrap(np.arctan2(Q, I)) 
            
            color = self._compute_color(val, sweep_min, sweep_max, sweep_cmap)
            axs['mag'].plot(freqs, mlog, color=color)
            axs['phase'].plot(freqs, phase, color=color) 

        return fig, axs

    def plot_mag(self,
            in_group='base', frequency_bound=None, inds=None,
            sweep_param=None, sweep_cmap='viridis', sweep_label=None,
        ): 
        """ Plot the magnitude for a specified group in the hierarchy. """
        in_group = '/' + in_group.strip('/')
        data_group = f"{in_group}/data"
        rg_arr, rgi_arr, rr_arr, start_inds = self._get_index_arrays(data_group)

        # - apply indices --------------- #
        if inds is None:
            inds = np.arange(start_inds.shape[0])
            
        if sweep_param is None:
            sweep_param_vals = np.arange(start_inds.shape[0])
            param = 'iter' 
        else:
            subgroup, param = sweep_param.split('.') 
            sweep_group_path = f"{in_group}/{subgroup}"
            sweep_param_vals = self[sweep_group_path][param].values[inds]
            
        sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max() 

        # - configure figure and axes objects - #
        fig, axs = self._configure_subplot_mosaic(
            [['mag']], sweep_param_vals, width_ratios=[0.95, 0.05],
            sweep_label=sweep_label, sweep_cmap=sweep_cmap,
        )
        axs['mag'].set(xlabel='Frequency (GHz.)', ylabel=self.mag_ylabel) 

        # - plot - #
        for i, val in zip(inds, sweep_param_vals):
            data = self._get_group_values(data_group, i, frequency_bound=frequency_bound)
            I, Q, freqs = data.I.values, data.Q.values, data.frequency.values 
            freqs *= 1e-9 
            mlog = (1 + self.power*1)*10*np.log10(np.sqrt(I**2 + Q**2)) 
            color = self._compute_color(val, sweep_min, sweep_max, sweep_cmap)
            axs['mag'].plot(freqs, mlog, color=color)

        return fig, axs

    def plot_phase(self,
            in_group='base', frequency_bound=None, inds=None,
            sweep_param=None, sweep_cmap='viridis', sweep_label=None,
        ): 
        """ Plot the phase for a specified group in the hierarchy. """
        in_group = '/' + in_group.strip('/')
        data_group = f"{in_group}/data"
        rg_arr, rgi_arr, rr_arr, start_inds = self._get_index_arrays(data_group)

        # - apply indices --------------- #
        if inds is None:
            inds = np.arange(start_inds.shape[0])
            
        if sweep_param is None:
            sweep_param_vals = np.arange(start_inds.shape[0])
            param = 'iter' 
        else:
            subgroup, param = sweep_param.split('.') 
            sweep_group_path = f"{in_group}/{subgroup}"
            sweep_param_vals = self[sweep_group_path][param].values[inds]
            
        sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max() 

        # - configure figure and axes objects - #
        fig, axs = self._configure_subplot_mosaic(
            [['phase']], sweep_param_vals, width_ratios=[0.95, 0.05],
            sweep_label=sweep_label, sweep_cmap=sweep_cmap,
        )
        axs['phase'].set(xlabel='Frequency (GHz.)', ylabel=self.phase_ylabel) 

        # - plot - #
        for i, val in zip(inds, sweep_param_vals):
            data = self._get_group_values(data_group, i, frequency_bound=frequency_bound)
            I, Q, freqs = data.I.values, data.Q.values, data.frequency.values 
            freqs *= 1e-9 
            phase = np.unwrap(np.arctan2(Q, I)) 
            color = self._compute_color(val, sweep_min, sweep_max, sweep_cmap)
            axs['phase'].plot(freqs, phase, color=color)

        return fig, axs

    def plot_iq(self,
            in_group='base', frequency_bound=None, inds=None,
            sweep_param=None, sweep_cmap='viridis', sweep_label=None,
        ):
        """ Plot IQ data for a specified group in the hierarchy. """
        in_group = '/' + in_group.strip('/')
        data_group = f"{in_group}/data"
        rg_arr, rgi_arr, rr_arr, start_inds = self._get_index_arrays(data_group)

        # - apply indices --------------- #
        if inds is None:
            inds = np.arange(start_inds.shape[0])
            
        if sweep_param is None:
            sweep_param_vals = np.arange(start_inds.shape[0])
            param = 'iter' 
        else:
            subgroup, param = sweep_param.split('.') 
            sweep_group_path = f"{in_group}/{subgroup}"
            sweep_param_vals = self[sweep_group_path][param].values[inds]
            
        sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max() 

        # - configure figure and axes objects - #
        fig, axs = self._configure_subplot_mosaic(
            [['iq']], sweep_param_vals, width_ratios=[0.95, 0.05],
            sweep_label=sweep_label, sweep_cmap=sweep_cmap,
        )
        axs['iq'].set(xlabel='I', ylabel='Q')

        # - plot - #
        for i, val in zip(inds, sweep_param_vals):
            color = self._compute_color(val, sweep_min, sweep_max, sweep_cmap) 
            data = self._get_group_values(data_group, i, frequency_bound=frequency_bound)
            I, Q, freqs = data.I.values, data.Q.values, data.frequency.values 
            axs['iq'].scatter(I, Q, marker='.', color=color)

        return fig, axs

    def plot_res_params(self, 
            in_group='base', inds=None, xparam=None, xparam_label=None, 
            frequency_scale='GHz.', frequency_offset=0, params_name='res_params'
        ):
        """ Plot the fit resonator parameters on a single summary figure. """
        in_group = '/' + in_group.strip('/')
        data_group = f"{in_group}/data"
        params_group = f"{in_group}/{params_name}"
        
        rg_arr, rgi_arr, rr_arr, start_inds = self._get_index_arrays(data_group)
        
        if inds is None:
            inds = np.arange(start_inds.shape[0])

        # - configure plot - # 
        fig, axs = plt.subplot_mosaic(
            [['fr', 'Ql'], 
             ['Q', 'Q']],
        )
        axs['fr'].set(
            xlabel='' if xparam_label is None else xparam_label,
            ylabel=r'$f_r$ (%s)' % frequency_scale,
        )
        axs['Ql'].set(
            xlabel='' if xparam_label is None else xparam_label,
            ylabel=r'$Q_l$', 
        )
        axs['Q'].set(
            xlabel='' if xparam_label is None else xparam_label,
            ylabel=r'$Q$', 
        )

        # - extract resonator parameters and x sweep value - #
        res_params = self._get_group_values(params_group, inds, index_group=data_group) 
        fr = res_params.fr.values
        ql = res_params.Ql.values
        qi, qc = res_params.Qi.values, res_params.Qc.values.real 
        
        # - check for error columns - #
        fr_err = res_params.fr_err.values if 'fr_err' in res_params.columns else None
        ql_err = res_params.Ql_err.values if 'Ql_err' in res_params.columns else None
        qi_err = res_params.Qi_err.values if 'Qi_err' in res_params.columns else None
        qc_err = res_params.Qc_err.values if 'Qc_err' in res_params.columns else None
        
        x_err = None
        if xparam is None: 
            x = np.arange(res_params.shape[0])[inds]
        else:
            subgroup, param = xparam.split('.') 
            sweep_group_path = f"{in_group}/{subgroup}"
            group_df = self._get_group_values(sweep_group_path, inds, index_group=data_group) 
            x = group_df[param].values
            
            # Check for x error column
            if f"{param}_err" in group_df.columns:
                x_err = group_df[f"{param}_err"].values

        # - formatting logic - #
        fr_multiply = {
            'GHz.': 1e-9,
            'MHz.': 1e-6,
            'kHz.': 1e-3,
            'Hz.': 1,
        }[frequency_scale]
        
        # Standard error scales multiplicatively, but ignores constant offsets
        if fr_err is not None:
            fr_err = fr_err * fr_multiply

        # - plot - #
        axs['fr'].errorbar(x, (fr-frequency_offset)*fr_multiply, xerr=x_err, yerr=fr_err, fmt='o')
        axs['Ql'].errorbar(x, ql, xerr=x_err, yerr=ql_err, fmt='o')
        axs['Q'].errorbar(x, qi, xerr=x_err, yerr=qi_err, fmt='o', label=r'$Q_i$')
        axs['Q'].errorbar(x, qc, xerr=x_err, yerr=qc_err, fmt='o', label=r'$Q_c$')
        axs['Q'].legend()

        return fig, axs

    def plot_params(self, in_group, param_x, param_y, param_x_label=None, param_y_label=None, scatter=True, plot_kwargs=None):
        """ Plot one or more parameters on a y axis against a single parameter on an x axis. """
        if plot_kwargs is None:
            plot_kwargs = {}
            
        in_group = '/' + in_group.strip('/')
        
        x_subgroup, x_param = param_x.split('.') 
        y_subgroup, y_param = param_y.split('.') 
        
        x_path = f"{in_group}/{x_subgroup}"
        y_path = f"{in_group}/{y_subgroup}"
        
        xvals = self[x_path][x_param].values 
        yvals = self[y_path][y_param].values

        fig, ax = plt.subplots()
        ax.set(
            xlabel=param_x if param_x_label is None else param_x_label,
            ylabel=param_y if param_y_label is None else param_y_label,
        )

        if scatter:
            ax.scatter(xvals, yvals, **plot_kwargs)
        else:
            ax.plot(xvals, yvals, **plot_kwargs)

        return fig, ax

# - RESONATOR PARAMETER FITTING -------------------------------------------------------------- #
    def fit_res_params(self,
            in_group, frequency_bound=None, inds=None, plot=False, plot_text=False, 
            phase_fit_kwargs=None, fixed_Qc=None, sweep_param=None, 
            sweep_cmap='viridis', sweep_label=None, 
        ):
        """ Fit resonator parameters. This writes purely to `in_group/res_params`. """
        
        in_group = '/' + in_group.strip('/')
        data_group = f"{in_group}/data"
        tmp_out_prefix = '/tmp_fit_process'
        
        keys = self.keys()
        for k in keys:
            if k.startswith(tmp_out_prefix): self.remove(k)

        rg_arr, rgi_arr, rr_arr, start_inds = self._get_index_arrays(data_group)

        if inds is None:
            inds = np.arange(start_inds.shape[0])
            
        if sweep_param is None:
            sweep_param_vals = np.arange(start_inds.shape[0])
            param = 'iter' 
        else:
            subgroup, param = sweep_param.split('.') 
            sweep_path = f"{in_group}/{subgroup}"
            sweep_param_vals = self[sweep_path][param].values[inds]
            
        sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max() 

        if plot:
            if plot_text:
                mosaic = [['iq', 'params'], ['centered_phase', 'centered_phase']]
            else:
                mosaic = [['iq'], ['centered_phase']] 
            fig, axs = self._configure_subplot_mosaic(
                mosaic, sweep_param_vals=sweep_param_vals, width_ratios=[0.95, 0.05],
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

        try: 
            for i, val in zip(inds, sweep_param_vals):
                ind = start_inds[i]
                rg_val, rgi_val = rg_arr[ind], rgi_arr[ind]
                
                data = self._get_group_values(data_group, i, frequency_bound=frequency_bound)
                I, Q, freqs = data.I.values, data.Q.values, data.frequency.values
                mlin = np.sqrt(I**2 + Q**2) 
                phase = np.unwrap(np.arctan2(Q, I)) 
                sdata = mlin*np.exp(1j*phase)
                
                xc, yc, r = circle_fit(sdata)
                Icentered = I - xc
                Qcentered = Q - yc
                centered_phase = np.unwrap(np.arctan2(Qcentered, Icentered))
                
                phase_fit_kwargs = {} if phase_fit_kwargs is None else phase_fit_kwargs 
                params, pcov = self._centered_phase_fit(freqs, centered_phase, **phase_fit_kwargs)
                theta0, Ql, fr = params
                
                phi = -np.arcsin(yc/r)
                if self.geometry == 'hanger': 
                    if fixed_Qc is None: 
                        Qc = Ql / (2*r*np.exp(-1j*phi))
                        Qcr = np.real(Qc)
                        Qi = 1 / ((1/Ql) - (1/Qcr))
                    else:
                        Qcr = fixed_Qc 
                        Qi = Qcr / (np.cos(phi) - 2*r)
                        Qc = Qcr + 1j*(Qi*Qcr*np.sin(phi) / (2*r*(Qi + Qcr)))
                elif self.geometry == 'shunt':
                    if fixed_Qc is None: 
                        Qc = 2*Ql / (2*r*np.exp(-1j*phi))
                        Qcr = np.real(Qc)
                        Qi = 1 / ((1/Ql) - (1/Qcr))
                    else:
                        Qcr = fixed_Qc 
                        Qi = Qcr / (np.cos(phi) - r)
                        Qc = Qcr + 1j*(Qi*Qcr*np.sin(phi) / (r*(Qi + Qcr)))
                        
                res_params_df = pd.DataFrame({'Ql': Ql, 'Qi': Qi, 'Qc': Qc, 'phi': phi, 'fr': fr}, 
                    index=pd.MultiIndex.from_product([[rg_val], [rgi_val]], names=['RecordGroup', 'RecordGroupInd']))
                self.append(f"{tmp_out_prefix}/res_params", res_params_df) 

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
                            r'$Q_l = %0.2f$' % Ql, r'$Q_i = %0.2f$' % Qi, r'$Q_{cr} = %0.2f$' % np.real(Qc),
                            r'$\phi = %0.2f$' % phi, r'$f_r = %0.2f$ (GHz.)' % (fr*1e-9)
                        ])
                        axs['params'].text(0.2, 0.2, params_str, fontsize=16)

            # - Only overwrite the res_params dataset in in_group - #
            target_key = f"{in_group}/res_params"
            if target_key in self.keys():
                self.remove(target_key)
                
            tmp_keys = [k for k in self.keys() if k.startswith(tmp_out_prefix)]
            for k in tmp_keys:
                sub_name = k.split('/')[-1]
                df = self.select(k)
                self.append(f"{in_group}/{sub_name}", df)
                self.remove(k)

        except Exception as e:
            for k in self.keys():
                if k.startswith(tmp_out_prefix): self.remove(k)
            raise e
        return ret


