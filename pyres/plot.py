import numpy as np
import matplotlib.pyplot as plt


def plot_mag_phase(
        store, in_group='base', inds=None, query=None,
        sweep_param=None, sweep_cmap='viridis', sweep_label=None,
        trace_label=None
    ):
    """
    Plot the magnitude and phase for a specified group in the hierarchy.
    
    :param store: The data store containing the resonator scattering data.
    :type store: ResonatorScatteringStore
    :param in_group: The group path from which to pull data (e.g., 'base'). Defaults to 'base'.
    :type in_group: str, optional
    :param inds: Record start indices to plot over. If None, all available data will be plotted.
    :type inds: numpy.ndarray or list, optional
    :param query: Optional pandas query string to filter traces and points (e.g., 'data.frequency <= 5e9').
    :type query: str, optional
    :param sweep_param: String formatted as 'subgroup.column' to map uniqueness over.
    :type sweep_param: str, optional
    :param sweep_cmap: Colormap used to indicate the value of the swept parameter. Defaults to 'viridis'.
    :type sweep_cmap: str, optional
    :param sweep_label: String label used to indicate the colorbar.
    :type sweep_label: str, optional
    :param trace_label: Optional list of string labels for each trace. If provided, replaces the colorbar with a legend.
    :type trace_label: list of str, optional
    :return: The generated matplotlib figure and axes mosaic.
    :rtype: tuple(matplotlib.figure.Figure, dict)
    """
    in_group = '/' + in_group.strip('/')
    data_group = f"{in_group}/data"
    
    rg_arr, rgi_arr, rr_arr, start_inds = store._get_index_arrays(data_group)

    if inds is None:
        inds = np.arange(start_inds.shape[0])
        
    point_masks = None
    if query is not None:
        inds, point_masks = store.group_query(in_group, query, inds)
        
    if trace_label is not None and len(trace_label) != len(inds):
        raise ValueError(f"Length of trace_label ({len(trace_label)}) must match the number of plotted traces ({len(inds)}).")
        
    if sweep_param is None:
        sweep_param_vals = np.arange(start_inds.shape[0])[inds] if len(inds) > 0 else np.array([])
        param = 'iter' 
    else:
        subgroup, param = sweep_param.split('.') 
        sweep_group_path = f"{in_group}/{subgroup}"
        sweep_param_vals = store[sweep_group_path][param].values[inds]
        
    if len(sweep_param_vals) > 0:
        sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max() 
    else:
        sweep_min, sweep_max = 0, 1

    if trace_label is not None:
        fig, axs = plt.subplot_mosaic([['mag'], ['phase']])
    else:
        fig, axs = store._configure_subplot_mosaic(
            [['mag'], ['phase']], sweep_param_vals, width_ratios=[0.95, 0.05],
            sweep_label=sweep_label, sweep_cmap=sweep_cmap,
        )
        
    axs['mag'].set_xticks([]) 
    axs['phase'].set_xlabel('Frequency (GHz)')
    axs['mag'].set_ylabel(store.mag_ylabel)
    axs['phase'].set_ylabel(store.phase_ylabel)

    for idx, (i, val) in enumerate(zip(inds, sweep_param_vals)):
        data = store._get_group_values(data_group, i)
        
        # Apply specific point masks if a trace-level query was used
        if point_masks is not None and i in point_masks:
            valid_rows = point_masks[i]
            if 'RecordRow' in data.index.names:
                mask = data.index.get_level_values('RecordRow').isin(valid_rows)
                data = data.loc[mask]
                
        if data.empty:
            continue
            
        I, Q, freqs = data.I.values, data.Q.values, data.frequency.values 
        freqs *= 1e-9 
        mlog = (1 + store.power*1)*10*np.log10(np.sqrt(I**2 + Q**2)) 
        phase = np.unwrap(np.arctan2(Q, I)) 
        
        color = store._compute_color(val, sweep_min, sweep_max, sweep_cmap)
        label = trace_label[idx] if trace_label is not None else None
        
        axs['mag'].plot(freqs, mlog, color=color, label=label)
        axs['phase'].plot(freqs, phase, color=color) # Only label 'mag' axis to avoid duplicate legend entries

    if trace_label is not None:
        fig.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout(rect=[0, 0, 0.85, 1])

    return fig, axs


def plot_mag(
        store, in_group='base', inds=None, query=None,
        sweep_param=None, sweep_cmap='viridis', sweep_label=None,
        trace_label=None
    ): 
    """
    Plot the magnitude for a specified group in the hierarchy.
    
    :param store: The data store containing the resonator scattering data.
    :type store: ResonatorScatteringStore
    :param in_group: The group path from which to pull data (e.g., 'base'). Defaults to 'base'.
    :type in_group: str, optional
    :param inds: Record start indices to plot over. If None, plots all available data.
    :type inds: numpy.ndarray or list, optional
    :param query: Optional pandas query string to filter traces and points (e.g., 'data.frequency <= 5e9').
    :type query: str, optional
    :param sweep_param: String formatted as 'subgroup.column' to map uniqueness over.
    :type sweep_param: str, optional
    :param sweep_cmap: Colormap used to indicate the value of the swept parameter. Defaults to 'viridis'.
    :type sweep_cmap: str, optional
    :param sweep_label: String label used to indicate the colorbar.
    :type sweep_label: str, optional
    :param trace_label: Optional list of string labels for each trace. If provided, replaces the colorbar with a legend.
    :type trace_label: list of str, optional
    :return: The generated matplotlib figure and axes mosaic.
    :rtype: tuple(matplotlib.figure.Figure, dict)
    """
    in_group = '/' + in_group.strip('/')
    data_group = f"{in_group}/data"
    rg_arr, rgi_arr, rr_arr, start_inds = store._get_index_arrays(data_group)

    if inds is None:
        inds = np.arange(start_inds.shape[0])
        
    point_masks = None
    if query is not None:
        inds, point_masks = store.group_query(in_group, query, inds)
        
    if trace_label is not None and len(trace_label) != len(inds):
        raise ValueError(f"Length of trace_label ({len(trace_label)}) must match the number of plotted traces ({len(inds)}).")
        
    if sweep_param is None:
        sweep_param_vals = np.arange(start_inds.shape[0])[inds] if len(inds) > 0 else np.array([])
        param = 'iter' 
    else:
        subgroup, param = sweep_param.split('.') 
        sweep_group_path = f"{in_group}/{subgroup}"
        sweep_param_vals = store[sweep_group_path][param].values[inds]
        
    if len(sweep_param_vals) > 0:
        sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max() 
    else:
        sweep_min, sweep_max = 0, 1

    if trace_label is not None:
        fig, axs = plt.subplot_mosaic([['mag']])
    else:
        fig, axs = store._configure_subplot_mosaic(
            [['mag']], sweep_param_vals, width_ratios=[0.95, 0.05],
            sweep_label=sweep_label, sweep_cmap=sweep_cmap,
        )
        
    axs['mag'].set(xlabel='Frequency (GHz)', ylabel=store.mag_ylabel) 

    for idx, (i, val) in enumerate(zip(inds, sweep_param_vals)):
        data = store._get_group_values(data_group, i)
        
        # Apply specific point masks if a trace-level query was used
        if point_masks is not None and i in point_masks:
            valid_rows = point_masks[i]
            if 'RecordRow' in data.index.names:
                mask = data.index.get_level_values('RecordRow').isin(valid_rows)
                data = data.loc[mask]
                
        if data.empty:
            continue
            
        I, Q, freqs = data.I.values, data.Q.values, data.frequency.values 
        freqs *= 1e-9 
        mlog = (1 + store.power*1)*10*np.log10(np.sqrt(I**2 + Q**2)) 
        color = store._compute_color(val, sweep_min, sweep_max, sweep_cmap)
        label = trace_label[idx] if trace_label is not None else None
        
        axs['mag'].plot(freqs, mlog, color=color, label=label)

    if trace_label is not None:
        fig.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout(rect=[0, 0, 0.85, 1])

    return fig, axs


def plot_phase(
        store, in_group='base', inds=None, query=None,
        sweep_param=None, sweep_cmap='viridis', sweep_label=None,
        trace_label=None
    ): 
    """
    Plot the phase for a specified group in the hierarchy.
    
    :param store: The data store containing the resonator scattering data.
    :type store: ResonatorScatteringStore
    :param in_group: The group path from which to pull data (e.g., 'base'). Defaults to 'base'.
    :type in_group: str, optional
    :param inds: Record start indices to plot over. If None, plots all available data.
    :type inds: numpy.ndarray or list, optional
    :param query: Optional pandas query string to filter traces and points (e.g., 'data.frequency <= 5e9').
    :type query: str, optional
    :param sweep_param: String formatted as 'subgroup.column' to map uniqueness over.
    :type sweep_param: str, optional
    :param sweep_cmap: Colormap used to indicate the value of the swept parameter. Defaults to 'viridis'.
    :type sweep_cmap: str, optional
    :param sweep_label: String label used to indicate the colorbar.
    :type sweep_label: str, optional
    :param trace_label: Optional list of string labels for each trace. If provided, replaces the colorbar with a legend.
    :type trace_label: list of str, optional
    :return: The generated matplotlib figure and axes mosaic.
    :rtype: tuple(matplotlib.figure.Figure, dict)
    """
    in_group = '/' + in_group.strip('/')
    data_group = f"{in_group}/data"
    rg_arr, rgi_arr, rr_arr, start_inds = store._get_index_arrays(data_group)

    if inds is None:
        inds = np.arange(start_inds.shape[0])
        
    point_masks = None
    if query is not None:
        inds, point_masks = store.group_query(in_group, query, inds)
        
    if trace_label is not None and len(trace_label) != len(inds):
        raise ValueError(f"Length of trace_label ({len(trace_label)}) must match the number of plotted traces ({len(inds)}).")
        
    if sweep_param is None:
        sweep_param_vals = np.arange(start_inds.shape[0])[inds] if len(inds) > 0 else np.array([])
        param = 'iter' 
    else:
        subgroup, param = sweep_param.split('.') 
        sweep_group_path = f"{in_group}/{subgroup}"
        sweep_param_vals = store[sweep_group_path][param].values[inds]
        
    if len(sweep_param_vals) > 0:
        sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max() 
    else:
        sweep_min, sweep_max = 0, 1

    if trace_label is not None:
        fig, axs = plt.subplot_mosaic([['phase']])
    else:
        fig, axs = store._configure_subplot_mosaic(
            [['phase']], sweep_param_vals, width_ratios=[0.95, 0.05],
            sweep_label=sweep_label, sweep_cmap=sweep_cmap,
        )
        
    axs['phase'].set(xlabel='Frequency (GHz)', ylabel=store.phase_ylabel) 

    for idx, (i, val) in enumerate(zip(inds, sweep_param_vals)):
        data = store._get_group_values(data_group, i)
        
        # Apply specific point masks if a trace-level query was used
        if point_masks is not None and i in point_masks:
            valid_rows = point_masks[i]
            if 'RecordRow' in data.index.names:
                mask = data.index.get_level_values('RecordRow').isin(valid_rows)
                data = data.loc[mask]
                
        if data.empty:
            continue
            
        I, Q, freqs = data.I.values, data.Q.values, data.frequency.values 
        freqs *= 1e-9 
        phase = np.unwrap(np.arctan2(Q, I)) 
        color = store._compute_color(val, sweep_min, sweep_max, sweep_cmap)
        label = trace_label[idx] if trace_label is not None else None
        
        axs['phase'].plot(freqs, phase, color=color, label=label)

    if trace_label is not None:
        fig.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout(rect=[0, 0, 0.85, 1])

    return fig, axs


def plot_iq(
        store, in_group='base', inds=None, query=None,
        sweep_param=None, sweep_cmap='viridis', sweep_label=None,
        trace_label=None
    ):
    """
    Plot IQ data for a specified group in the hierarchy on the complex plane.
    
    :param store: The data store containing the resonator scattering data.
    :type store: ResonatorScatteringStore
    :param in_group: The group path from which to pull data (e.g., 'base'). Defaults to 'base'.
    :type in_group: str, optional
    :param inds: Record start indices to plot over. If None, plots all available data.
    :type inds: numpy.ndarray or list, optional
    :param query: Optional pandas query string to filter traces and points (e.g., 'data.frequency <= 5e9').
    :type query: str, optional
    :param sweep_param: String formatted as 'subgroup.column' to map uniqueness over.
    :type sweep_param: str, optional
    :param sweep_cmap: Colormap used to indicate the value of the swept parameter. Defaults to 'viridis'.
    :type sweep_cmap: str, optional
    :param sweep_label: String label used to indicate the colorbar.
    :type sweep_label: str, optional
    :param trace_label: Optional list of string labels for each trace. If provided, replaces the colorbar with a legend.
    :type trace_label: list of str, optional
    :return: The generated matplotlib figure and axes mosaic.
    :rtype: tuple(matplotlib.figure.Figure, dict)
    """
    in_group = '/' + in_group.strip('/')
    data_group = f"{in_group}/data"
    rg_arr, rgi_arr, rr_arr, start_inds = store._get_index_arrays(data_group)

    if inds is None:
        inds = np.arange(start_inds.shape[0])
        
    point_masks = None
    if query is not None:
        inds, point_masks = store.group_query(in_group, query, inds)
        
    if trace_label is not None and len(trace_label) != len(inds):
        raise ValueError(f"Length of trace_label ({len(trace_label)}) must match the number of plotted traces ({len(inds)}).")
        
    if sweep_param is None:
        sweep_param_vals = np.arange(start_inds.shape[0])[inds] if len(inds) > 0 else np.array([])
        param = 'iter' 
    else:
        subgroup, param = sweep_param.split('.') 
        sweep_group_path = f"{in_group}/{subgroup}"
        sweep_param_vals = store[sweep_group_path][param].values[inds]
        
    if len(sweep_param_vals) > 0:
        sweep_min, sweep_max = sweep_param_vals.min(), sweep_param_vals.max() 
    else:
        sweep_min, sweep_max = 0, 1

    if trace_label is not None:
        fig, axs = plt.subplot_mosaic([['iq']])
        axs['iq'].set_aspect('equal')
    else:
        fig, axs = store._configure_subplot_mosaic(
            [['iq']], sweep_param_vals, width_ratios=[0.95, 0.05],
            sweep_label=sweep_label, sweep_cmap=sweep_cmap,
        )
        
    axs['iq'].set(xlabel='I', ylabel='Q')

    for idx, (i, val) in enumerate(zip(inds, sweep_param_vals)):
        data = store._get_group_values(data_group, i)
        
        # Apply specific point masks if a trace-level query was used
        if point_masks is not None and i in point_masks:
            valid_rows = point_masks[i]
            if 'RecordRow' in data.index.names:
                mask = data.index.get_level_values('RecordRow').isin(valid_rows)
                data = data.loc[mask]
                
        if data.empty:
            continue
            
        color = store._compute_color(val, sweep_min, sweep_max, sweep_cmap) 
        label = trace_label[idx] if trace_label is not None else None
        
        I, Q = data.I.values, data.Q.values
        axs['iq'].scatter(I, Q, marker='.', color=color, label=label)

    if trace_label is not None:
        fig.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout(rect=[0, 0, 0.85, 1])

    return fig, axs


def plot_res_params(
        store, in_group='base', inds=None, query=None, xparam=None, xparam_label=None, 
        frequency_scale='GHz', frequency_offset=0, params_name='res_params'
    ):
    """
    Plot the fit resonator parameters on a single summary figure.
    
    :param store: The data store containing the fitted parameters.
    :type store: ResonatorScatteringStore
    :param in_group: The group path from which to pull parameter data. Defaults to 'base'.
    :type in_group: str, optional
    :param inds: Record start indices to plot over. If None, plots all available data.
    :type inds: numpy.ndarray or list, optional
    :param query: Optional pandas query string to filter traces (e.g., 'meta.start_frequency <= 5e9').
    :type query: str, optional
    :param xparam: String formatted as 'subgroup.column' to use for the x-axis.
    :type xparam: str, optional
    :param xparam_label: String label to apply to the x-axis.
    :type xparam_label: str, optional
    :param frequency_scale: Unit scalar for the frequency axis ('GHz', 'MHz', 'kHz', 'Hz'). Defaults to 'GHz'.
    :type frequency_scale: str, optional
    :param frequency_offset: Offset to subtract from the frequency before scaling. Defaults to 0.
    :type frequency_offset: float, optional
    :param params_name: Name of the subgroup containing the fit parameters. Defaults to 'res_params'.
    :type params_name: str, optional
    :return: The generated matplotlib figure and axes mosaic.
    :rtype: tuple(matplotlib.figure.Figure, dict)
    """
    in_group = '/' + in_group.strip('/')
    data_group = f"{in_group}/data"
    params_group = f"{in_group}/{params_name}"
    
    rg_arr, rgi_arr, rr_arr, start_inds = store._get_index_arrays(data_group)
    
    if inds is None:
        inds = np.arange(start_inds.shape[0])
        
    if query is not None:
        inds, _ = store.group_query(in_group, query, inds)

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
    
    if len(inds) == 0:
        return fig, axs

    res_params = store._get_group_values(params_group, inds, index_group=data_group) 
    fr = res_params.fr.values
    ql = res_params.Ql.values
    qi, qc = res_params.Qi.values, res_params.Qc.values.real 
    
    fr_err = res_params.fr_err.values if 'fr_err' in res_params.columns else None
    ql_err = res_params.Ql_err.values if 'Ql_err' in res_params.columns else None
    qi_err = res_params.Qi_err.values if 'Qi_err' in res_params.columns else None
    qc_err = res_params.Qc_err.values if 'Qc_err' in res_params.columns else None
    
    x_err = None
    if xparam is None: 
        x = np.arange(res_params.shape[0])
    else:
        subgroup, param = xparam.split('.') 
        sweep_group_path = f"{in_group}/{subgroup}"
        group_df = store._get_group_values(sweep_group_path, inds, index_group=data_group) 
        x = group_df[param].values
        
        if f"{param}_err" in group_df.columns:
            x_err = group_df[f"{param}_err"].values

    fr_multiply = {
        'GHz': 1e-9,
        'MHz': 1e-6,
        'kHz': 1e-3,
        'Hz': 1,
    }[frequency_scale]
    
    if fr_err is not None:
        fr_err = fr_err * fr_multiply

    axs['fr'].errorbar(x, (fr-frequency_offset)*fr_multiply, xerr=x_err, yerr=fr_err, fmt='o')
    axs['Ql'].errorbar(x, ql, xerr=x_err, yerr=ql_err, fmt='o')
    axs['Q'].errorbar(x, qi, xerr=x_err, yerr=qi_err, fmt='o', label=r'$Q_i$')
    axs['Q'].errorbar(x, qc, xerr=x_err, yerr=qc_err, fmt='o', label=r'$Q_c$')
    axs['Q'].legend()

    return fig, axs


def plot_params(
        store, in_group, param_x, param_y, query=None, param_x_label=None, param_y_label=None, 
        scatter=True, plot_kwargs=None
    ):
    """
    Plot one or more parameters on a y-axis against a single parameter on an x-axis.
    
    :param store: The data store containing the parameters to plot.
    :type store: ResonatorScatteringStore
    :param in_group: The group path from which to pull data.
    :type in_group: str
    :param param_x: String formatted as 'subgroup.column' to use for the x-axis.
    :type param_x: str
    :param param_y: String formatted as 'subgroup.column' to use for the y-axis.
    :type param_y: str
    :param query: Optional pandas query string to filter traces (e.g., 'meta.start_frequency <= 5e9').
    :type query: str, optional
    :param param_x_label: String label to apply to the x-axis. If None, defaults to param_x.
    :type param_x_label: str, optional
    :param param_y_label: String label to apply to the y-axis. If None, defaults to param_y.
    :type param_y_label: str, optional
    :param scatter: If True, uses a scatter plot. If False, uses a line plot. Defaults to True.
    :type scatter: bool, optional
    :param plot_kwargs: Additional keyword arguments to pass to the matplotlib plot function.
    :type plot_kwargs: dict, optional
    :return: The generated matplotlib figure and axes.
    :rtype: tuple(matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    if plot_kwargs is None:
        plot_kwargs = {}
        
    in_group = '/' + in_group.strip('/')
    data_group = f"{in_group}/data"
    
    # Establish a consistent baseline inds array based on the primary index group
    rg_arr, rgi_arr, rr_arr, start_inds = store._get_index_arrays(data_group)
    inds = np.arange(start_inds.shape[0])
    
    if query is not None:
        inds, _ = store.group_query(in_group, query, inds)
        
    fig, ax = plt.subplots()
    ax.set(
        xlabel=param_x if param_x_label is None else param_x_label,
        ylabel=param_y if param_y_label is None else param_y_label,
    )

    if len(inds) == 0:
        return fig, ax
    
    x_subgroup, x_param = param_x.split('.') 
    y_subgroup, y_param = param_y.split('.') 
    
    x_path = f"{in_group}/{x_subgroup}"
    y_path = f"{in_group}/{y_subgroup}"
    
    # Safely fetch values strictly for the filtered inds
    xvals = store._get_group_values(x_path, inds, index_group=data_group)[x_param].values 
    yvals = store._get_group_values(y_path, inds, index_group=data_group)[y_param].values

    if scatter:
        ax.scatter(xvals, yvals, **plot_kwargs)
    else:
        ax.plot(xvals, yvals, **plot_kwargs)

    return fig, ax
