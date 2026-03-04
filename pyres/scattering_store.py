import numpy as np
from numpy.polynomial import Polynomial
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt


from .helpers import circle_fit


class ResonatorScatteringStore(pd.HDFStore):

    def __init__(self, path, geometry, power=True, **kwargs):
        super().__init__(path, mode='a', **kwargs)
        assert geometry in ['hanger', 'shunt'], 'Invalid geometry specified. The options are ["hanger", "shunt"]' 
        self.geometry = geometry 
        self.data_storer = self.get_storer('data')
        self.rg = self.data_storer.read_column('RecordGroup')
        self.rgi = self.data_storer.read_column('RecordGroupInd')
        self.rr = self.data_storer.read_column('RecordRow')
        self.record_start_inds = np.where(self.rr == '000000')[0]
        self.power = power
        self.sparam = '21' if (self.geometry == 'hanger') else '11'
        self.mag_ylabel = r'$|S_{%s}|$' % self.sparam 
        self.phase_ylabel = r'$\angle S_{%s}$ (rads)' % self.sparam 

    def _get_group_values(self, group, index, param=None, frequency_bound=None):
        """ Return the dataframe from the group at the specified index.

        :param group: String corresponding to the HDF group to pull the dataframe from.
        :param index: Index from the RecordGroup and RecordGroupInd list to pull dataframe from.
        :param param: Parameter to pull from the dataframe. 
        :param frequency_bound: Frequency limits to take HDF group data between. 
        """
        ind = self.record_start_inds[index] 
        rg, rgi = self.rg[ind], self.rgi[ind] 
        where_str = f'RecordGroup == "{rg}" & RecordGroupInd == "{rgi}"'
        df = self.select(group, where=where_str)
        if frequency_bound is not None and 'frequency' in df.columns:
            freqs = df.frequency.values 
            inds = (frequency_bound[0] < freqs) * (freqs < frequency_bound[1])
            df = df.iloc[inds]
        if param is not None:
            ret = df[param].values
        else:
            ret = df

        return ret

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
    def _line_func(freqs, tau, offset):
        return -2*np.pi*tau*freqs + offset

    @staticmethod 
    def _centered_phase_func(freqs, theta0, Ql, fr):
        return theta0 + 2*np.arctan(2*Ql*(1 - (freqs/fr)))

    def _centered_phase_fit(self, freqs, phase_data, **kwargs):
        params, pcov = spopt.curve_fit(self._centered_phase_func, freqs, phase_data, **kwargs) 

        return params, pcov

    # - CALIBRATION FUNCTIONS -------------------------------------------------------------------- #
    def calibrate_cable_delay(self, 
            tau=None, offset=None, 
            frequency_bound=None, fit_frequency_bound=None, inds=None, cal=False, 
            plot=False, sweep_param=None, sweep_cmap='viridis', sweep_label=None,
        ):
        """ Remove a line from the unwrapped phase data.
        
        :param tau: Fixed cable delay slope. If None a line will be fit to the unwrapped phase
                    between the frequency bounds.
        :param offset: Fixed cable delay offset. Behaves the same as above.
        :param frequency_bound: Frequency range over which calibration should be performed. 
        :param fit_frequency_bound: Frequency range over which a line fit should be performed. 
        :param inds: Indices over which to perform the calibration. 
        :param cal: Boolean to indicate if the existing calibration data should be used. 
        :param plot: Boolean to indicate if a plot showing the calibration results should be generated. 
        :param sweep_param: String to indicate a parameter that is swept over in the data.
        :param sweep_cmap: Colormap to use to indicate the value of the swept parameter.
        :param sweep_label: String label used to label the colorbar. 
        """
        # - apply indices ------------------------------------------------------------- # 
        if inds is None:
            inds = np.arange(self.record_start_inds.shape[0])

        # - set up sweep parameter and plot ------------------------------------------- #
        # - sweep parameter - # 
        if sweep_param is None:
            sweep_param_vals = np.arange(self.record_start_inds.shape[0])
            param='iter' 
        else:
            group, param = sweep_param.split('.') 
            sweep_param_vals = self[group][param].values
        sweep_min = sweep_param_vals.min()
        sweep_max = sweep_param_vals.max()
        fig, axs = None, None 
        if plot:
            fig, axs = self._configure_subplot_mosaic(
                [['phase_raw', 'iq_raw'], ['phase_cal', 'iq_cal']],
                sweep_param_vals,
                width_ratios=[0.45, 0.45, 0.1],
                sweep_label=sweep_label,
                sweep_cmap=sweep_cmap,
            )
            axs['phase_cal'].set(
                xlabel='Frequency (GHz.)',
                ylabel=self.phase_ylabel,
            )
            axs['phase_raw'].set(
                xlabel='Frequency (GHz.)', 
                ylabel=self.phase_ylabel
            )
            axs['iq_cal'].set(
                ylabel='Q',
                xlabel='I'
            )
            axs['iq_raw'].set(
                ylabel='Q',
                xlabel='I',
            )
            ret = fig, axs
        else:
            ret = None

        # - remove line from the data ------------------------------------------------- # 
        for i, val in zip(inds, sweep_param_vals): 
            ind = self.record_start_inds[i] 
            rg, rgi = self.rg[ind], self.rgi[ind] 
            if param != 'iter':
                sweep_val = self._get_group_values(group, i, param=param, frequency_bound=frequency_bound)
            else:
                sweep_val = val 
            data_group = 'data' if not cal else 'cal_data'
            data = self._get_group_values(data_group, i, frequency_bound=frequency_bound) 
            freqs = data.frequency.values  
            I, Q = data.I.values, data.Q.values
            phase = np.unwrap(np.arctan2(Q, I))
            mlin = np.sqrt(I**2 + Q**2)
            if fit_frequency_bound is not None:
                inds = (fit_frequency_bound[0] < freqs) * (freqs < fit_frequency_bound[1])
                fit_freqs = freqs[inds]
                fit_phase = phase[inds]
                fit_mlin = mlin[inds]
            else:
                fit_freqs = freqs
                fit_phase = phase
                fit_mlin = mlin
            if (tau is None) and (offset is None):
                fit_func = self._line_func
                popt, pcov = spopt.curve_fit(fit_func, fit_freqs, fit_phase)
                tau_fit, offset_fit = popt
            elif (tau is None) and (offset is not None):
                fit_func = lambda freqs, tau: self._line_func(freqs, tau, offset)
                popt, pcov = spopt.curve_fit(fit_func, fit_freqs, fit_phase)
                tau_fit = popt[0]
                offset_fit = offset 
            elif (tau is not None) and (offset is None):
                fit_func = lambda freqs, offset: self._line_func(freqs, tau, offset)
                popt, pcov = spopt.curve_fit(fit_func, fit_freqs, fit_phase)
                tau_fit = tau 
                offset_fit = popt[1] 
            else: # - tau is not None and offset is not None
                tau_fit = tau
                offset_fit = offset 
            line = self._line_func(freqs, tau_fit, offset_fit)
            corrected_phase = phase - line
            Ical, Qcal = mlin*np.cos(corrected_phase), mlin*np.sin(corrected_phase)
            # - write to store - # 
            cal_df = pd.DataFrame({
                'frequency': freqs, 
                'I': Ical,
                'Q': Qcal,
            }, index=pd.MultiIndex.from_product(
                [[rg], [rgi], ['%06i' % j for j in np.arange(freqs.shape[0])]],
                names=['RecordGroup', 'RecordGroupInd', 'RecordRow'] 
                )
            )
            self.append('temp_data', cal_df) 
            params_df = pd.DataFrame({
                'tau': tau_fit,
                'cable_delay_offset': offset_fit,
            }, index=pd.MultiIndex.from_product(
                [[rg], [rgi]],
                names=['RecordGroup', 'RecordGroupInd']
                )
            ) 
            self.append('temp_params', params_df) 
            # - plotting - #
            if plot:
                plot_freqs = freqs*1e-9
                color = self._compute_color(val, sweep_min, sweep_max, sweep_cmap) 
                axs['phase_raw'].plot(plot_freqs, phase, color=color)
                axs['phase_raw'].plot(plot_freqs, line, ls=':', color='black')
                axs['phase_cal'].plot(plot_freqs, corrected_phase, color=color)
                axs['iq_raw'].scatter(I, Q, color=color, marker='.')
                axs['iq_cal'].scatter(Ical, Qcal, color=color, marker='.')

        if '/cal_data' in self.keys():
            self.remove('/cal_data')
        self.get_node('/temp_data')._f_rename('cal_data') 

        if '/cable_delay_params' in self.keys():
            self.remove('/cable_delay_params') 
        self.get_node('/temp_params')._f_rename('cable_delay_params')

        return ret

    def calibrate_constant_scaling(self,
            a=None, alpha=None, phase_fit_kwargs=None,
            inds=None, plot=False, cal=False, frequency_bound=None,
            sweep_param=None, sweep_cmap='viridis', sweep_label=None, 
        ):
        """ Calibrate a constant environmental attenuation and phase using the Probst method.
        
        :param a: Fixed attenuation/amplification factor.
        :param alpha: Fixed phase shift factor.
        :param phase_fit_kwargs: Keyword arguments to pass to the phase calibration fitting function. 
        :param inds: Record start indices to calibrate over.
        :param plot: Boolean to indicate if the calibration process should be plotted.
        :param cal: Boolean to indicate if data from the 'cal_data' group should be used. 
        :param sweep_param: String to indicate a swept parameter for a colorbar.
        :param sweep_cmap: Colormap used to indicate the value of the swept parameter.
        :param sweep_label: String label used to indicate the colorbar. 
        """ 
        # - apply indices --------------- #
        if inds is None:
            inds = np.arange(self.record_start_inds.shape[0])
        if sweep_param is None:
            sweep_param_vals = np.arange(self.record_start_inds.shape[0])
            param='iter' 
        else:
            group, param = sweep_param.split('.') 
            sweep_param_vals = self[group][param].values
        sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max() 

        if plot:
            fig, axs = self._configure_subplot_mosaic(
                [['iq_raw', 'iq_process'], ['centered_phase', 'iq_final']],
                sweep_param_vals=sweep_param_vals,
                width_ratios=[0.45, 0.45, 0.1],
                sweep_label=sweep_label,
                sweep_cmap=sweep_cmap,
            )
            for key, ax in axs.items():
                if 'iq' in key:
                    ax.set(
                        xlabel='I',
                        ylabel='Q'
                    )
            axs['centered_phase'].set(
                xlabel='Frequency (GHz.)',
                ylabel=self.mag_ylabel,
            ) 
            ret = fig, axs
        else:
            ret = None

        # - perform environmental calibration - # 
        for i, val in zip(inds, sweep_param_vals):
            ind = self.record_start_inds[i]
            rg, rgi = self.rg[ind], self.rgi[ind]
            try:
                iter(frequency_bound)
            except TypeError:
                fb = frequency_bound
            else:
                fb = frequency_bound[i] 
            if param != 'iter':
                sweep_val = self._get_group_values(group, i, param=param, frequency_bound=fb)
            else:
                sweep_val = val 
            data_group = 'data' if not cal else 'cal_data'
            # - extract data to fit - # 
            data = self._get_group_values(data_group, i, frequency_bound=fb)
            I, Q, freqs = data.I.values, data.Q.values, data.frequency.values
            mlin = np.sqrt(I**2 + Q**2) 
            phase = np.unwrap(np.arctan2(Q, I)) 
            sdata = mlin*np.exp(1j*phase)

            if plot:
                color = self._compute_color(sweep_val, sweep_min, sweep_max, sweep_cmap)
                plot_freqs = freqs*1e-9 
                axs['iq_raw'].scatter(I, Q, marker='.', color=color)

            if (a is None) or (alpha is None): 
                # - fit a circle, translate to the center - #
                xc, yc, r = circle_fit(sdata)
                Icentered = I - xc
                Qcentered = Q - yc
                centered_phase = np.unwrap(np.arctan2(Qcentered, Icentered))

                # - run a phase fit on the translated circle - #
                phase_fit_kwargs = {} if phase_fit_kwargs is None else phase_fit_kwargs 
                params, pcov = self._centered_phase_fit(
                    freqs, centered_phase,
                    **phase_fit_kwargs
                )
                theta0, Ql, fr = params

                # - compute off resonant point, constant environmental scaling, and phase shift - #
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

            # - remove environmental scaling from the data, write to store in a new group 'cal_data' - #
            a_cal = afit if a is None else a
            alpha_cal = alphafit if alpha is None else alpha
            factor = a_cal*np.exp(1j*alpha_cal)
            sdata /= factor
            cal_I, cal_Q = np.real(sdata), np.imag(sdata) 

            if plot:
                axs['iq_final'].scatter(cal_I, cal_Q, marker='.', color=color)

            # - write to store - #
            cal_df = pd.DataFrame({
                'frequency': freqs, 
                'I': cal_I,
                'Q': cal_Q,
            }, index=pd.MultiIndex.from_product(
                [[rg], [rgi], ['%06i' % j for j in np.arange(freqs.shape[0])]],
                names=['RecordGroup', 'RecordGroupInd', 'RecordRow'] 
                )
            )
            self.append('temp_data', cal_df) 
            cal_params_df = pd.DataFrame({
                'a': a_cal,
                'alpha': alpha_cal,
            }, index=pd.MultiIndex.from_product(
                [[rg], [rgi]],
                names=['RecordGroup', 'RecordGroupInd'],
                )
            )
            self.append('temp_params', cal_params_df)

        if '/cal_data' in self.keys():
            self.remove('/cal_data')
        self.get_node('/temp_data')._f_rename('cal_data')

        if '/constant_scaling_params' in self.keys():
            self.remove('/constant_scaling_params')
        self.get_node('/temp_params')._f_rename('constant_scaling_params')

        return ret

    def calibrate_polymag_background(self,
            frequency_bound=None, lower_frequency_bound=None, upper_frequency_bound=None, 
            inds=None, degree=2, fixed_coeffs=None, domain=None,
            cal=False, plot=False, sweep_param=None, 
            sweep_cmap='viridis', sweep_label=None, 
        ):
        """ Fit and remove a polynomial background from the magnitude data.
        
        :param frequency_bound: Global frequency limits for polynomial fitting. 
        :param lower_frequency_bound: Lower frequency limits on fitting a polynomial to magnitude data.
        :param upper_frequency_bound: Upper frequency limits for fitting a polynomial to magnitude data. 
        :param inds: Record start indices to plot over. If None, all available data will be plotted.
        :param degree: Degree of the polynomial to fit.
        :param fixed_coeffs: List of polynomial coefficients to apply a fixed calibration. 
        :param domain: Polynomial fit domain used to apply a fixed cablibration. 
        :param cal: Boolean to indicate if existing cal_data should be used.
        :param plot: Boolean to indicate if the calibration results should be plotted. 
        :param sweep_param: String to indicate a swept parameter for a colorbar.
        :param sweep_cmap: Colormap used to indicate the value of the swept parameter.
        :param sweep_label: String label used to indicate the colorbar.
        """
        # - apply indices --------------- #
        if inds is None:
            inds = np.arange(self.record_start_inds.shape[0])
        if sweep_param is None:
            sweep_param_vals = np.arange(self.record_start_inds.shape[0])
            param='iter' 
        else:
            group, param = sweep_param.split('.') 
            sweep_param_vals = self[group][param].values
        sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max() 

        if plot:
            fig, axs = self._configure_subplot_mosaic(
                [['mag_raw'], ['mag_cal']],
                sweep_param_vals,
                width_ratios=[0.9, 0.1],
                sweep_label=sweep_label,
                sweep_cmap=sweep_cmap,
            )
            axs['mag_raw'].set(
                xlabel='Frequency (GHz.)',
                ylabel=self.mag_ylabel,
            )
            axs['mag_cal'].set(
                xlabel='Frequency (GHz.)',
                ylabel=self.mag_ylabel,
            )
            ret = fig, axs
        else:
            ret = None

        # - apply background polynomial fitting and removal - #
        for i, val in zip(inds, sweep_param_vals):
            ind = self.record_start_inds[i]
            rg, rgi = self.rg[ind], self.rgi[ind] 
            if param != 'iter':
                sweep_val = self._get_group_values(group, i, param=param, frequency_bound=frequency_bound)
            else:
                sweep_val = val 
            data_group = 'data' if not cal else 'cal_data'
            # - extract data to fit - # 
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
                else: # - (upper_frequency_bound is None) and (lower_frequency_bound is None)
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

            # - remove the background - #
            fit_mlog = np.zeros_like(freqs)
            fit_mlog[(fit[0].min() <= freqs) * (freqs <= fit[0].max())] = fit[1](fit[0]) 
            cal_mlog = mlog - fit_mlog
            cal_mlin = 10**(cal_mlog / ((1 + 1*self.power)*10))
            cal_I, cal_Q = cal_mlin*np.cos(phase), cal_mlin*np.sin(phase)

            # - write to store - # 
            cal_df = pd.DataFrame({
                'frequency': freqs, 
                'I': cal_I,
                'Q': cal_Q,
            }, index=pd.MultiIndex.from_product(
                [[rg], [rgi], ['%06i' % j for j in np.arange(freqs.shape[0])]],
                names=['RecordGroup', 'RecordGroupInd', 'RecordRow'] 
                )
            )
            self.append('temp_data', cal_df)
            cal_params_dict = {
                'x%i' % j: fit[1].coef[j] 
                for j in range(fit[1].coef.shape[0])
            } 
            cal_params_dict['domain_min'] = fit[1].domain.min()
            cal_params_dict['domain_max'] = fit[1].domain.max() 
            cal_params_df = pd.DataFrame(cal_params_dict, index=pd.MultiIndex.from_product(
                [[rg], [rgi]],
                names=['RecordGroup', 'RecordGroupInd'] 
                )
            )
            self.append('temp_params', cal_params_df)

            # - plot - #
            if plot:
                plot_freqs = freqs*1e-9 
                color = self._compute_color(sweep_val, sweep_min, sweep_max, sweep_cmap) 
                axs['mag_raw'].plot(plot_freqs, mlog, color=color)
                axs['mag_raw'].plot(plot_freqs, fit_mlog, ls=':', color='black') 
                axs['mag_cal'].plot(plot_freqs, cal_mlog, color=color)

        if '/cal_data' in self.keys():
            self.remove('/cal_data') 
        self.get_node('/temp_data')._f_rename('cal_data')

        if '/polymag_params' in self.keys():
            self.remove('/polymag_params')
        self.get_node('/temp_params')._f_rename('polymag_params')

        return ret

    def calibrate_polyphase_background(self,
            frequency_bound=None, lower_frequency_bound=None, upper_frequency_bound=None,
            inds=None, degree=2, fixed_coeffs=None, domain=None, cal=False, plot=False, 
            sweep_param=None, sweep_cmap='viridis', sweep_label=None,
        ):
        """ Fit and remove a polynomial background from the phase data.
        
        :param frequency_bound: Global frequency limits for polynomial fitting. 
        :param lower_frequency_bound: Lower frequency limits on fitting a polynomial to phase data.
        :param upper_frequency_bound: Upper frequency limits for fitting a polynomial to phase data. 
        :param inds: Record start indices to plot over. If None, all available data will be plotted.
        :param degree: Degree of the polynomial to fit.
        :param fixed_coeffs: List of polynomial coefficients to apply a fixed calibration. 
        :param sep: Boolean to indicate if the polynomial fit should be connected across the the lower and
        upper frequency bounds.
        :param cal: Boolean to indicate if existing cal_data should be used.
        :param plot: Boolean to indicate if the calibration results should be plotted. 
        :param sweep_param: String to indicate a swept parameter for a colorbar.
        :param sweep_cmap: Colormap used to indicate the value of the swept parameter.
        :param sweep_label: String label used to indicate the colorbar.
        """
        # - apply indices --------------- #
        if inds is None:
            inds = np.arange(self.record_start_inds.shape[0])
        if sweep_param is None:
            sweep_param_vals = np.arange(self.record_start_inds.shape[0])
            param='iter' 
        else:
            group, param = sweep_param.split('.') 
            sweep_param_vals = self[group][param].values
        sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max() 

        if plot:
            fig, axs = self._configure_subplot_mosaic(
                [['phase_raw'], ['phase_cal']],
                sweep_param_vals,
                width_ratios=[0.9, 0.1],
                sweep_label=sweep_label,
                sweep_cmap=sweep_cmap,
            )
            axs['phase_raw'].set(
                xlabel='Frequency (GHz.)',
                ylabel=self.phase_ylabel,
            )
            axs['phase_cal'].set(
                xlabel='Frequency (GHz.)',
                ylabel=self.phase_ylabel,
            )
            ret = fig, axs
        else:
            ret = None

        # - apply background polynomial fitting and removal - #
        for i, val in zip(inds, sweep_param_vals):
            ind = self.record_start_inds[i]
            rg, rgi = self.rg[ind], self.rgi[ind] 
            if param != 'iter':
                sweep_val = self._get_group_values(group, i, param=param, frequency_bound=frequency_bound)
            else:
                sweep_val = val 
            data_group = 'data' if not cal else 'cal_data'
            # - extract data to fit - # 
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
                else: # - (upper_frequency_bound is None) and (lower_frequency_bound is None)
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

            # - remove the background - #
            fit_phase = np.zeros_like(freqs)
            fit_phase[(fit[0].min() <= freqs) * (freqs <= fit[0].max())] = fit[1](fit[0])
            cal_phase = phase - fit_phase
            cal_I, cal_Q = mlin*np.cos(cal_phase), mlin*np.sin(cal_phase)

            # - write to store - # 
            cal_df = pd.DataFrame({
                'frequency': freqs, 
                'I': cal_I,
                'Q': cal_Q,
            }, index=pd.MultiIndex.from_product(
                [[rg], [rgi], ['%06i' % j for j in np.arange(freqs.shape[0])]],
                names=['RecordGroup', 'RecordGroupInd', 'RecordRow'] 
                )
            )
            self.append('temp_data', cal_df)
            cal_params_dict = {
                'x%i' % j: fit[1].coef[j]
                for j in range(fit[1].coef.shape[0])
            } 
            cal_params_dict['domain_min'] = fit[1].domain.min()
            cal_params_dict['domain_max'] = fit[1].domain.max() 
            cal_params_df = pd.DataFrame(cal_params_dict, index=pd.MultiIndex.from_product(
                [[rg], [rgi]],
                names=['RecordGroup', 'RecordGroupInd'] 
                )
            )
            self.append('temp_params', cal_params_df)

            # - plot - #
            if plot:
                color = self._compute_color(sweep_val, sweep_min, sweep_max, sweep_cmap) 
                axs['phase_raw'].plot(freqs, phase, color=color)
                axs['phase_raw'].plot(freqs, fit_phase, ls=':', color='black') 
                axs['phase_cal'].plot(freqs, cal_phase, color=color)

        if '/cal_data' in self.keys():
            self.remove('/cal_data') 
        self.get_node('/temp_data')._f_rename('cal_data')

        if '/polyphase_params' in self.keys():
            self.remove('/polyphase_params')
        self.get_node('/temp_params')._f_rename('polyphase_params')

        return ret

    def calibrate_from_file(self, filepath,
            frequency_bound=None, inds=None, plot=False,
            sweep_param=None, sweep_cmap='viridis', sweep_label=None, 
        ):
        """ Perform background calibration using a measured data file.

        :param filepath: String filepath to where the background datafile is stored. 
        :param frequency_bound: Limited frequency range over which to perform the calibration.
        :param inds: Record start indices to calibrate.
        :param plot: Boolean to indicate if calibration results should be plotted.
        :param sweep_param: String to indicate a swept parameter for a colorbar.
        :param sweep_cmap: Colormap used to indicate the value of the swept parameter.
        :param sweep_label: String label used to indicate the colorbar.
        """ 
        # - open background calibration data - # 
        bg_store = pd.HDFStore(filepath) 
        bg_data = bg_store.data
        bg_I, bg_Q, bg_freqs = bg_data.I.values, bg_data.Q.values, bg_data.frequency.values 
        if frequency_bound is not None:
            bg_inds = np.where((frequency_bound[0] <= bg_freqs) * (bg_freqs <= frequency_bound[1]))[0] 
            bg_I, bg_Q, bg_freqs = bg_I[bg_inds], bg_Q[bg_inds], bg_freqs[bg_inds] 
        bg_mlin = np.sqrt(bg_I**2 + bg_Q**2) 
        bg_mlog = (1 + 1*self.power)*10*np.log10(bg_mlin) 
        bg_phase = np.unwrap(np.arctan2(bg_Q, bg_I))

        # - apply indices - #
        if inds is None:
            inds = np.arange(self.record_start_inds.shape[0])
        if sweep_param is None:
            sweep_param_vals = np.arange(self.record_start_inds.shape[0])
            param='iter' 
        else:
            group, param = sweep_param.split('.') 
            sweep_param_vals = self[group][param].values
        sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max() 

        if plot:
            fig, axs = self._configure_subplot_mosaic(
                [['raw_mag', 'cal_mag'], ['raw_phase', 'cal_phase']],
                width_ratios=[0.45, 0.45, 0.1],
                sweep_cmap=sweep_cmap,
                sweep_label=sweep_label,
                sweep_param_vals=sweep_param_vals,
            )
            for key, ax in axs.items():
                ax.set_xlabel('Frequency (GHz.)')
                if 'mag' in key:
                    ax.set_ylabel(self.mag_ylabel)
                else:
                    ax.set_ylabel(self.phase_ylabel)
            ret = fig, axs
        else:
            ret = None

        background_plotted = False 
        for i, val in zip(inds, sweep_param_vals):
            ind = self.record_start_inds[i]
            rg, rgi = self.rg[ind], self.rgi[ind]
            if param != 'iter':
                sweep_val = self._get_group_values(group, i, param=param, frequency_bound=frequency_bound)
            else:
                sweep_val = val
            data = self._get_group_values('data', i, frequency_bound=frequency_bound)
            I, Q, freqs = data.I.values, data.Q.values, data.frequency.values
            mlin = np.sqrt(I**2 + Q**2)
            mlog = (1 + 1*self.power)*10*np.log10(mlin) 
            phase = np.unwrap(np.arctan2(Q, I))
            sdata = mlin*np.exp(1j*phase)
            
            # - interpolate the calibration data to the same frequency values as the measurement data - # 
            bg_mlin_interp = np.interp(freqs, bg_freqs, bg_mlin) 
            bg_mlog_interp = (1 + 1*self.power)*10*np.log10(bg_mlin_interp) 
            bg_phase_interp = np.interp(freqs, bg_freqs, bg_phase) 
            bg_sdata_interp = bg_mlin_interp*np.exp(1j*bg_phase_interp) 

            # - divide out cal data, write back to the store - #
            cal_sdata = sdata / bg_sdata_interp 
            cal_I, cal_Q = np.real(cal_sdata), np.imag(cal_sdata)
            cal_mlog = (1 + 1*self.power)*10*np.log10(np.sqrt(cal_I**2 + cal_Q**2))  
            cal_phase = np.unwrap(np.arctan2(cal_Q, cal_I)) 
            cal_df = pd.DataFrame({
                'frequency': freqs,
                'I': cal_I,
                'Q': cal_Q, 
            }, index=pd.MultiIndex.from_product(
                [[rg], [rgi], ['%06i' % j for j in np.arange(freqs.shape[0])]],
                names=['RecordGroup', 'RecordGroupInd', 'RecordRow'] 
                )
            )
            self.append('temp_data', cal_df) 

            # - plot - #
            if plot:
                plot_freqs = freqs*1e-9 
                color = self._compute_color(sweep_val, sweep_min, sweep_max, sweep_cmap) 
                if not background_plotted:
                    axs['raw_mag'].plot(plot_freqs, bg_mlog_interp, ls=':', color='black')
                    axs['raw_phase'].plot(plot_freqs, bg_phase_interp, ls=':', color='black')
                    background_plotted = True
                axs['raw_mag'].plot(plot_freqs, mlog, color=color) 
                axs['raw_phase'].plot(plot_freqs, phase, color=color) 
                axs['cal_mag'].plot(plot_freqs, cal_mlog, color=color)
                axs['cal_phase'].plot(plot_freqs, cal_phase, color=color)

        if '/cal_data' in self.keys():
            self.remove('/cal_data') 
        self.get_node('/temp_data')._f_rename('cal_data')

        return ret

    # - RESONATOR PARAMETER FITTING -------------------------------------------------------------- #
    def fit_res_params(self,
            frequency_bound=None, inds=None, plot=False, plot_text=False, phase_fit_kwargs=None, fixed_Qc=None,
            cal=True, sweep_param=None, sweep_cmap='viridis', sweep_label=None, 
        ):
        """ Fit resonator parameters to the calibrated IQ data.

        :param frequency_bound: Frequency range over which to plot.
        :param inds: Record start indices to plot over. If None, all available data will be plotted.
        :param plot: Boolean to indicate if the circle fit results should be plotted. 
        :param plot_text: Add text with the resonator fit parameters. Only really useful for single traces at the moment.  
        :param phase_fit_kwargs: Dictionary with keyword arguments for the phase fitting.
        :param fixed_Qc: Optional fixed value of the coupling Q. 
        :param sweep_param: String to indicate a swept parameter for a colorbar.
        :param sweep_cmap: Colormap used to indicate the value of the swept parameter.
        :param sweep_label: String label used to indicate the colorbar.
        """ 
        # - apply indices --------------- #
        if inds is None:
            inds = np.arange(self.record_start_inds.shape[0])
        if sweep_param is None:
            sweep_param_vals = np.arange(self.record_start_inds.shape[0])
            param='iter' 
        else:
            group, param = sweep_param.split('.') 
            sweep_param_vals = self[group][param].values
        sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max() 

        if plot:
            if plot_text:
                mosaic = [['iq', 'params'], ['centered_phase', 'centered_phase']]
            else:
                mosaic = [['iq'], ['centered_phase']] 
            fig, axs = self._configure_subplot_mosaic(
                mosaic,
                sweep_param_vals=sweep_param_vals,
                width_ratios=[0.9, 0.1],
                sweep_label=sweep_label,
                sweep_cmap=sweep_cmap,
            )
            axs['iq'].set(
                xlabel='I',
                ylabel='Q',
            )
            axs['centered_phase'].set(
                xlabel='Frequency (GHz.)',
                ylabel=self.phase_ylabel,
            )
            if plot_text:
                axs['params'].set_xticks([])
                axs['params'].set_yticks([])
                for key, spine in axs['params'].spines.items():
                    spine.set_visible(False)
            ret = fig, axs
        else:
            ret = None

        # - fit 
        data_group = 'data' if not cal else 'cal_data' 
        for i, val in zip(inds, sweep_param_vals):
            ind = self.record_start_inds[i]
            rg, rgi = self.rg[ind], self.rgi[ind]
            if param != 'iter':
                sweep_val = self._get_group_values(group, i, param=param, frequency_bound=frequency_bound)
            else:
                sweep_val = val
            data = self._get_group_values(data_group, i, frequency_bound=frequency_bound)
            I, Q, freqs = data.I.values, data.Q.values, data.frequency.values
            mlin = np.sqrt(I**2 + Q**2) 
            phase = np.unwrap(np.arctan2(Q, I)) 
            sdata = mlin*np.exp(1j*phase)
            
            # - fit a circle, translate to the center - # 
            xc, yc, r = circle_fit(sdata)
            Icentered = I - xc
            Qcentered = Q - yc
            centered_phase = np.unwrap(np.arctan2(Qcentered, Icentered))
            
            # - run a phase fit on the translated circle - #
            phase_fit_kwargs = {} if phase_fit_kwargs is None else phase_fit_kwargs 
            params, pcov = self._centered_phase_fit(
                freqs, centered_phase,
                **phase_fit_kwargs
            )
            theta0, Ql, fr = params
            
            # - extract the resonator parameters, write them to the store - # 
            phi = -np.arcsin(yc/r)
            if self.geometry == 'hanger': 
                if fixed_Qc is None: 
                    Qc = Ql / (2*r*np.exp(-1j*phi))
                    Qcr = np.real(Qc)
                    Qi_inv = (1/Ql) - (1/Qcr)
                    Qi = 1 / Qi_inv
                else:
                    Qcr = fixed_Qc 
                    Qi = Qcr / (np.cos(phi) - 2*r)
                    Qci = Qi*Qcr*np.sin(phi) / (2*r*(Qi + Qcr))
                    Qc = Qcr + 1j*Qci
            elif self.geometry == 'shunt':
                if fixed_Qc is None: 
                    Qc = 2*Ql / (2*r*np.exp(-1j*phi))
                    Qcr = np.real(Qc)
                    Qi_inv = (1/Ql) - (1/Qcr)
                    Qi = 1 / Qi_inv
                else:
                    Qcr = fixed_Qc 
                    Qi = Qcr / (np.cos(phi) - r)
                    Qci = Qi*Qcr*np.sin(phi) / (r*(Qi + Qcr))
                    Qc = Qcr + 1j*Qci
            index = pd.MultiIndex.from_product([[rg], [rgi]], names=['RecordGroup', 'RecordGroupInd'])
            res_params_df = pd.DataFrame({
                'Ql': Ql,
                'Qi': Qi,
                'Qc': Qc,
                'phi': phi,
                'fr': fr,
            }, index=index)
            self.append('temp_params', res_params_df) 

            # - plot - #
            if plot:
                plot_freqs = freqs * 1e-9
                phase_fit = self._centered_phase_func(freqs, theta0, Ql, fr) 
                color = self._compute_color(sweep_val, sweep_min, sweep_max, sweep_cmap)
                axs['iq'].scatter(I, Q, marker='.', color=color)
                circle = plt.Circle((xc, yc), r, edgecolor='r', facecolor='none', linewidth=2)
                axs['iq'].add_patch(circle)
                axs['centered_phase'].plot(plot_freqs, centered_phase, color=color)
                axs['centered_phase'].plot(plot_freqs, phase_fit, ls=':', color='black')
                if plot_text:
                    params_str = '\n'.join([
                        r'$Q_l = %0.2f$' % Ql,
                        r'$Q_i = %0.2f$' % Qi,
                        r'$Q_{cr} = %0.2f$' % np.real(Qc),
                        r'$\phi = %0.2f$' % phi,
                        r'$f_r = %0.2f$ (GHz.)' % (fr*1e-9)
                    ])
                    axs['params'].text(0.2, 0.2, params_str, fontsize=16)

        if '/res_params' in self.keys():
            self.remove('/res_params')
        self.get_node('/temp_params')._f_rename('res_params')

        return ret

    # - PLOTTING FUNCTIONS ----------------------------------------------------------------------- # 
    def plot_mag_phase(self,
            frequency_bound=None, inds=None,
            sweep_param=None, sweep_cmap='viridis', sweep_label=None,          
        ):
        # - apply indices --------------- #
        if inds is None:
            inds = np.arange(self.record_start_inds.shape[0])
        if sweep_param is None:
            sweep_param_vals = np.arange(self.record_start_inds.shape[0])
            param='iter' 
        else:
            group, param = sweep_param.split('.') 
            sweep_param_vals = self[group][param].values
        sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max() 

        # - configure figure and axes objects - #
        if '/cal_data' not in self.keys():
            fig, axs = self._configure_subplot_mosaic(
                [['mag_raw'], ['phase_raw']],
                sweep_param_vals,
                width_ratios=[0.9, 0.1],
                sweep_label=sweep_label,
                sweep_cmap=sweep_cmap,
            )
        else:
            fig, axs = self._configure_subplot_mosaic(
                [['mag_raw', 'mag_cal'], ['phase_raw', 'phase_cal']],
                sweep_param_vals,
                width_ratios=[0.45, 0.45, 0.1],
                sweep_label=sweep_label,
                sweep_cmap=sweep_cmap,    
            )
            axs['mag_cal'].set(
               xticks=[],
            )
            axs['phase_cal'].set(
                xlabel='Frequency (GHz.)',
            )
        axs['mag_raw'].set_xticks([]) 
        axs['phase_raw'].set_xlabel('Frequency (GHz.)')
        axs['mag_raw'].set_ylabel(self.mag_ylabel)
        axs['phase_raw'].set_ylabel(self.phase_ylabel)

        # - plot - #
        for i, val in zip(inds, sweep_param_vals):
            data = self._get_group_values('data', i, frequency_bound=frequency_bound)
            I, Q, freqs = data.I.values, data.Q.values, data.frequency.values 
            freqs *= 1e-9 
            mlog = (1 + self.power*1)*10*np.log10(np.sqrt(I**2 + Q**2)) 
            phase = np.unwrap(np.arctan2(Q, I)) 
            color = self._compute_color(val, sweep_min, sweep_max, sweep_cmap)
            axs['mag_raw'].plot(freqs, mlog, color=color)
            axs['phase_raw'].plot(freqs, phase, color=color) 
            if '/cal_data' in self.keys():
                cal_data = self._get_group_values('cal_data', i)
                Ical, Qcal, freqs_cal = cal_data.I.values, cal_data.Q.values, cal_data.frequency.values 
                freqs_cal *= 1e-9 
                mlog_cal = (1 + self.power*1)*10*np.log10(np.sqrt(Ical**2 + Qcal**2)) 
                phase_cal = np.unwrap(np.arctan2(Qcal, Ical)) 
                axs['mag_cal'].plot(freqs_cal, mlog_cal, color=color)
                axs['phase_cal'].plot(freqs_cal, phase_cal, color=color)

        return fig, axs

    def plot_iq(self,
        frequency_bound=None, inds=None,
        sweep_param=None, cal_sweep_param=None, 
        sweep_cmap='viridis', sweep_label=None, 
    ):
        """ Plot IQ data.

        :param frequency_bound: Frequency range over which to plot.
        :param inds: Record start indices to plot over. If None, all available data will be plotted.
        :param sweep_param: String to indicate a swept parameter for a colorbar.
        :param cal_sweep_param: String to indicate an alternative sweep parameter for calibration data. 
        :param sweep_cmap: Colormap used to indicate the value of the swept parameter.
        :param sweep_label: String label used to indicate the colorbar.
        """ 
        # - apply indices --------------- #
        if inds is None:
            inds = np.arange(self.record_start_inds.shape[0])
        if sweep_param is None:
            sweep_param_vals = np.arange(self.record_start_inds.shape[0])
            param='iter' 
        else:
            group, param = sweep_param.split('.') 
            sweep_param_vals = self[group][param].values
        sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max() 

        # - configure figure and axes objects - #
        if '/cal_data' not in self.keys():
            fig, axs = self._configure_subplot_mosaic(
                [['iq_raw']],
                sweep_param_vals,
                width_ratios=[0.9, 0.1],
                sweep_label=sweep_label,
                sweep_cmap=sweep_cmap,
            )
        else:
            fig, axs = self._configure_subplot_mosaic(
                [['iq_raw'], ['iq_cal']],
                sweep_param_vals,
                width_ratios=[0.9, 0.1],
                sweep_label=sweep_label,
                sweep_cmap=sweep_cmap,
            )
            axs['iq_cal'].set(
                xlabel='I',
                ylabel='Q'
            )
        axs['iq_raw'].set(
            xlabel='I',
            ylabel='Q',
        )

        # - plot - #
        for i, val in zip(inds, sweep_param_vals):
            if param != 'iter':
                sweep_val = self._get_group_values(group, i, param=param, frequency_bound=frequency_bound)
            else:
                sweep_val = val 
            color = self._compute_color(sweep_val, sweep_min, sweep_max, sweep_cmap) 
            data = self._get_group_values('data', i, frequency_bound=frequency_bound)
            I, Q, freqs = data.I.values, data.Q.values, data.frequency.values 
            freqs *= 1e-9 
            axs['iq_raw'].scatter(I, Q, marker='.', color=color)
            if '/cal_data' in self.keys(): 
                cal_data = self._get_group_values('cal_data', i)
                Ical, Qcal, freqs_cal = cal_data.I.values, cal_data.Q.values, cal_data.frequency.values 
                freqs_cal *= 1e-9 
                if cal_sweep_param is not None: 
                    group, param = cal_sweep_param.split('.')
                    cal_sweep_val = self._get_group_values(group, i, param=param, frequency_bound=frequency_bound)
                    cal_color = self._compute_color(cal_sweep_val, sweep_min, sweep_max, sweep_cmap) 
                else:
                    cal_color = color 
                axs['iq_cal'].scatter(Ical, Qcal, marker='.', color=cal_color)

        return fig, axs

    def plot_params(self, param_x, param_y, param_x_label=None, param_y_label=None, scatter=True, plot_kwargs={}):
        """ Plot one or more parameters on a y axis against a single parameter on an x axis.
        
        :param param_x: Parameter to plot on the x-axis.
        :param param_y: Parameter to plot on the y-axis. Single string or list-like.
        :param param_x_label: String label to use for the plot x-axis.
        :param param_y_label: String label to use for the plot y-axis. 
        :param scatter: Boolean to indicate if a scatter plot format should be used. 
        """
        xgroup, xparam = param_x.split('.') 
        xvals = self[xgroup][xparam].values 
        ygroup, yparam = param_y.split('.') 
        xvals, yvals = self[xgroup][xparam].values, self[ygroup][yparam].values

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

    def plot_res_params(self, 
            inds=None, xparam=None, xparam_label=None, 
            frequency_bound=None, x_cmap='viridis',
            data_group='cal_data',
        ):
        """ Plot the fit resonator parameters on a single summary figure.
        """
        if inds is None:
            inds = np.arange(self.record_start_inds.shape[0])
        if xparam is None:
            xparam_vals = np.arange(self.record_start_inds.shape[0])
            xparam='iter' 
        else:
            group, param = xparam.split('.') 
            xparam_vals = self[group][param].values
        xmin, xmax = xparam_vals.min(), xparam_vals.max() 
        
        # - configure plot - # 
        fig, axs = plt.subplot_mosaic(
            [['fr', 'Ql'], 
             ['Q', 'Q']],
        )
        axs['fr'].set(
            xlabel='' if xparam_label is None else xparam_label,
            ylabel=r'$f_r$ (GHz.)',
        )
        # axs['IQ'].set(
        #     xlabel=r'$\mathcal{R}\{S_{%s}\}$' % self.sparam,
        #     ylabel=r'$\mathcal{I}\{S_{%s}\}$' % self.sparam,
        #     aspect='equal',
        # )
        axs['Ql'].set(
            xlabel='' if xparam_label is None else xparam_label,
            ylabel=r'$Q_l$', 
        )
        axs['Q'].set(
            xlabel='' if xparam_label is None else xparam_label,
            ylabel=r'$Q$', 
        )

        # - extract resonator parameters and x sweep value - # 
        res_params = self.res_params
        fr = res_params.fr.values
        ql = res_params.Ql.values
        qi, qc = res_params.Qi.values, res_params.Qc.values.real 
        if xparam == 'iter':
            x = np.arange(res_params.shape[0])
        else:
            xparam_split = xparam.split('.') 
            group, param = xparam_split 
            x = self[group][param].values

        # - plot - #
        axs['fr'].scatter(
            x, fr*1e-9, marker='.' 
        )
        axs['Ql'].scatter(
            x, ql, marker='.'
        )
        axs['Q'].scatter(
            x, qi, marker='.', label=r'$Q_i$'
        )
        axs['Q'].scatter(
            x, qc, marker='.', label=r'$Q_c$'
        )
        axs['Q'].legend()

        return fig, axs
        