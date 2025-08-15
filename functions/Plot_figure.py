import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from . import Processing
from . import Cov
from . import Skill
from speccy import sick_tricks as gary
from speccy import utils as ut
import string
import pandas as pd
import seaborn as sns

def Model_fit_result(F_obs_list,P_obs_list, 
                    F_model_fit_list, P_model_fit_list,):
    num_subplots = len(F_obs_list)
    rows = 2
    cols = int(np.ceil(num_subplots/rows))
    fig, axes = plt.subplots(rows, cols,sharex=True,sharey=True)
    axes = axes.flatten()
    # fig.text(0.5, 0.99, 'Model fit result', ha='center', va='center')
    for i in range(len(F_obs_list)):
        #plot obs
        axes[i].plot(F_obs_list[i],P_obs_list[i],label='Residual Subset',alpha=0.5)
        #plot model fit
        axes[i].plot(F_model_fit_list[i],P_model_fit_list[i],
                     label='Mode {} Model Fit'.format(i+1),linewidth=2,alpha=0.8)
        axes[i].set_xscale("log")
        axes[i].set_yscale("log")
        axes[i].legend(loc='lower left')
        axes[i].set_xlim(0.5,40)
        # axes[i].set_ylim(1e-3,1e3)
    for ax in axes[-cols:]:
        ax.set_xlabel('Frequency [cpd]')
    # Dynamically set the ylabel only for left column subplots
    for ax in axes[::cols]:
        ax.set_ylabel('Wave Spectrum')
    plt.tight_layout()  # Adjust layout so titles and labels don't overlap
    # plt.show()   
    return fig, axes
    
def Plot_prediction_list(x, y_list, label, color='tab:red', alpha=0.05, ax=None):
    for order, i in enumerate(y_list):
        if ax is None:
            if order == 0:
                plt.plot(x, i, color=color, alpha=alpha, label=label)
            else:
                plt.plot(x, i, color=color, alpha=alpha)
        else:
            if order == 0:
                ax.plot(x, i, color=color, alpha=alpha, label=label)
            else:
                ax.plot(x, i, color=color, alpha=alpha)

def Plot_Density_fit_performance(ds_dict, bins =10):
    #compute the median of the desity fit
    mode_fraction_median_dict = {}
    for i in ds_dict:
        # median  = np.nanmedian(Processing.Cal_density_fit_percentage(ds_dict[i]))
        # median  = np.nanmean(Processing.Cal_density_fit_percentage(ds_dict[i]))
        median  = Processing.Cal_density_fit_percentage(ds_dict[i]).quantile(0.80).values
        if median >= 100:
            print('{}({}%) is overfitting'.format(i,np.round(median,2)))
        elif median <=50:
            print('{}({}%) is underfit'.format(i,np.round(median,2)))
        mode_fraction_median_dict[i] = median
    
    plt.hist(mode_fraction_median_dict.values(), bins = bins, color="skyblue", edgecolor="black")
    plt.xlabel("Density fit performance (%)")
    plt.ylabel("Frequency")
    plt.title('{} Modes'.format(len(ds_dict[i].modes)))
    plt.tight_layout()
    plt.show()

    return mode_fraction_median_dict
  
def Plot_IW_amp_spectrum(time_list,A_list,nmodes):
    row = nmodes
    column = 2
    fig, axes = plt.subplots(row, column,sharex='col',)  # 
    axes = axes.reshape(row, column)
    for mode_number in range(nmodes):
        # Plot time series on the left (first column)
        axes[mode_number, 0].plot(time_list[mode_number], A_list[mode_number])
        axes[mode_number, 0].set_title(f"Mode {mode_number + 1} Time Series")
        axes[mode_number, 0].set_ylabel('Displacement (m)')

        # Compute the spectrum
        Δ = (time_list[mode_number][1]-time_list[mode_number][0]).astype('float')/1e9/86400
        Δ = Δ.values
        F_obs,P_obs = Processing.Cal_periodogram(A_list[mode_number].values,Δ)
        # Plot spectrum on the right (second column)
        axes[mode_number, 1].plot(F_obs, P_obs)
        axes[mode_number, 1].set_title(f"Mode {mode_number + 1} Spectrum")
        axes[mode_number, 1].set_xscale('log')
        axes[mode_number, 1].set_yscale('log')
        axes[mode_number, 1].set_xlabel('Frequency (cpd)')
        axes[mode_number, 1].set_ylabel('PSD (m^2/cpd)')

    # Rotate x-axis labels for better readability
    for ax in axes[:, 0]:  # Time series column
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()  # Adjust layout so titles and labels don't overlap
    plt.show()

def Plot_HA_result(time_list,A_list,xcoords,
                   Mean_params_list,ϵ_list,Yd_mean_list,
                   F_ϵ_list,Puu_ϵ_list, mode_number=0):
    
    x = time_list[mode_number]
    y = A_list[mode_number]
    ϵ = ϵ_list[mode_number]
    yd_mean = Yd_mean_list[mode_number]
    #calculate spectrum
    Δ = (x[1]-x[0]).astype('float')/1e9/86400
    Δ = Δ.values
    F_obs,P_obs = Processing.Cal_periodogram(y.values,Δ)

    plt.subplot(2, 1, 1)
    idx = 15000
    plt.plot(x[10000:idx],y[10000:idx],label='Obs',alpha=0.5)
    plt.plot(x[10000:idx],ϵ[10000:idx],'-.',label = 'Residual')
    plt.plot(x[10000:idx],yd_mean[10000:idx],label='HA',linewidth=2.5)
    plt.xlabel('days')
    plt.ylabel('Amp (m)')
    # plt.grid(b=True,ls=':')
    plt.title('Time series mode {}'.format(mode_number+1))
    plt.legend(loc="lower right")

    F_ϵ   = F_ϵ_list[mode_number]
    Puu_ϵ = Puu_ϵ_list[mode_number]
    Peaks = Processing.Coherent_peaks(xcoords[1:],Mean_params_list[mode_number],F_ϵ)

    plt.subplot(2, 1, 2)
    plt.plot(F_obs,P_obs,label='Obs',alpha=0.5)
    plt.plot(F_ϵ,Puu_ϵ,label='Residual')
    plt.plot(F_ϵ,Peaks,label='HA',linewidth=2.5)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('f_mean[cycles per day]')
    plt.ylabel('PSD(m²/cpd)')
    # plt.grid(b=True,ls=':')
    plt.legend(loc="lower right") 
    plt.ylim(1e-5, 1e3)
    plt.xlim(0.4,50)

def Plot_each_component_model_fit(df,dict_name,
                                  mode = 3, model_type = 'M1P1'):
    #pre-setup
    n = 600000
    delta = 600/86400
    tt = ut.taus(n, delta)
    #select the df
    selected_df = df[(df['Dict_name'] == dict_name) & (df['Mode'] == mode)]
    if model_type == 'M1P1':
        print(selected_df.loc[:, 'η_c':'γ_D2'])
        params = selected_df.loc[:, 'η_c':'γ_D2'].values[0]
        
        acf_true_model  = Cov.M1P1(tt, params)
        acf_true_LR2    = Cov.LR_2(tt, params[2:],l_cos=1.965)
    elif model_type == 'M1P2': 
        params = selected_df.loc[:, 'η_c':'γ_M2'].values[0]
        acf_true_model = Cov.M1P2_2(tt, params)
        acf_true_LR2 = Cov.LR_2(tt, params[2:], l_cos=1.93) \
                     + Cov.LR_2(tt, params[5:], l_cos=1.99)   #M2+S2
    else:
        raise ValueError("model_type must be 'M1P2' or 'M1P1'")
    acf_true_Matern = Cov.Matern(tt, params,lmbda=3)
    print(selected_df)
    #numerically calculate spectrum from acf
    ff_model, S_bias_model = gary.bochner(acf_true_model, delta, bias=True)
    ff_LR2, S_bias_LR2     = gary.bochner(acf_true_LR2, delta, bias=True)
    ff_Matern, S_bias_Matern = gary.bochner(acf_true_Matern, delta, bias=True)
    #plot
    plt.plot(ff_LR2[ff_LR2>=0], S_bias_LR2[ff_LR2>=0], label="LR2", linestyle="-.",color='blue')  
    plt.plot(ff_Matern[ff_Matern>=0], S_bias_Matern[ff_Matern>=0], label="Matern", linestyle="-.",color='red')   
    plt.plot(ff_model[ff_model>=0], S_bias_model[ff_model>=0], label=model_type,color='black') 
    plt.title(dict_name)
    plt.xlim(0.5,220)
    # plt.ylabel("PSD (K²/cpd)")
    plt.xlabel("Frequency (cpd)")
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(fontsize = 20)

    return plt

def Plot_parameter_medium_sns(ax, df, parameter, parameter_unit,):
    # Filter groups with at least 2 data points
    valid_groups = df.groupby(['Site', 'Mode'], observed=True).filter(lambda g: len(g) >= 2)
    
    sns.pointplot(
        data=valid_groups,
        x="Site",
        y=parameter,
        hue="Mode",
        ax=ax,
        estimator="median",
        dodge=0.25,
        markers="o",
        capsize=0.1,
        errorbar=('pi', 50)
    )
    ax.set_ylabel(f"{parameter} ({parameter_unit})")
    ax.grid(True)

    # Remove subplot legends (we'll show one master legend later)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.tick_params(axis='x', rotation=45)


def Plot_parameter_value(ax, df, parameter, xlabel=False):
    # Iterate over sites and plot with offsets
    for i, (site, site_data) in enumerate(df.groupby('Site',observed=True)):
        # Apply an offset based on index 'i' to separate overlapping points
        offset = (i - len(df['Site'].unique()) / 2) * 0.05
        # Scatter plot for individual data points
        ax.scatter(
            site_data['Mode'] + offset,
            site_data[parameter],
            label=site, 
            marker='o',
            alpha=0.8)
        ax.grid(True)
    ax.set_xticks(df['Mode'].unique())
    # Conditionally set x-axis label
    if xlabel:
        ax.set_xlabel('Mode')


def Plot_seasonality_by_mode(df, site_to_plot, parameter_list, figsize=(6, 3)):
    # Filter by site
    df_site = df[df["Site"] == site_to_plot].copy()
    # Map seasons to numbers
    season_order = df_site["season"].unique()
    season_map = {s: i for i, s in enumerate(season_order)}
    df_site["season_num"] = df_site["season"].map(season_map)
    # Offset by mode
    mode_list = sorted(df_site["Mode"].unique())
    mode_offset = {m: (i - (len(mode_list)-1) / 2) * 0.1 for i, m in enumerate(mode_list)}
    # Create subplots
    fig, axes = plt.subplots(len(parameter_list), 1, 
                             figsize=(figsize[0], figsize[1] * len(parameter_list)), sharex=True)
    # Plot each parameter
    for i, (ax, param) in enumerate(zip(axes, parameter_list)):
        df_site["x"] = df_site.apply(lambda r: r["season_num"] + mode_offset[r["Mode"]], axis=1)
        sns.scatterplot(data=df_site, x="x", y=param, hue="Mode", ax=ax, s=80, palette="tab10")
        ax.set_title(f"{param} vs Season for {site_to_plot}")
        ax.set_ylabel(param)
        ax.set_xticks(list(season_map.values()))
        ax.set_xticklabels(list(season_map.keys()))

        legend = ax.get_legend()
        if i == 0:
            if legend:
                legend.set_title("Mode")
                legend.set_bbox_to_anchor((1.02, 1))
                legend.set_loc("upper left")
        else:
            if legend:
                legend.remove()
    axes[-1].set_xlabel("Season")
    plt.tight_layout()
    return fig, axes


#only works for variance parameter
def Plot_η_depth_variablity_with_mode(filtered_temp_dict,η_name,mode,colors,):
    cols = len(η_name)
    rows= 1
    fig, axes = plt.subplots(rows, cols,sharey=True)
    axes = axes.flatten()
    for parameter_order, parameter_name in enumerate(η_name):
        parameter_df_plot_median_all_mode = []
        parameter_df_plot_iqr_all_mode = []  
        for mode_number in mode:
            parameter_dict =  {key: variable for key, variable in filtered_temp_dict.items() if key[2] == parameter_name and key[1] == mode_number}
            # Create a list to hold the data for plotting
            parameter_df_plot = Processing.Transfer_to_plot_df(parameter_dict)
            parameter_df_plot_median = parameter_df_plot.groupby('depth')['values'].median()
            # Calculate the IQR (Q3 - Q1)
            parameter_df_plot_iqr = parameter_df_plot.groupby('depth')['values'].quantile(0.75) - parameter_df_plot.groupby('depth')['values'].quantile(0.25)
            # append to list
            parameter_df_plot_median_all_mode.append(parameter_df_plot_median)
            parameter_df_plot_iqr_all_mode.append(parameter_df_plot_iqr)
            #plot
            y_offset = 2
            axes[parameter_order].errorbar(parameter_df_plot_median,
                                           parameter_df_plot_median.index + mode_number * y_offset,
                                           xerr=parameter_df_plot_iqr, color=colors[mode_number],
                                           capsize=3, alpha=0.8)
            axes[parameter_order].set_xlabel('value (K)')
            
        #calculate the sum
        parameter_df_plot_median_sum = pd.DataFrame(sum(df for df in parameter_df_plot_median_all_mode))
        parameter_df_plot_iqr_sum = pd.DataFrame(sum(df ** 2 for df in parameter_df_plot_iqr_all_mode) ** 0.5)  # Summing variances
        # Plot the sum of modes using IQR for error bars
        axes[parameter_order].errorbar(parameter_df_plot_median_sum['values'],
                                       parameter_df_plot_median.index + (mode_number + 1) * y_offset,
                                       xerr=parameter_df_plot_iqr_sum['values'], color=colors[-1],
                                       capsize=3, label='Sum')
        axes[parameter_order].set_title(f'{parameter_name}')
        axes[parameter_order].grid(True)
        
     # Create a legend for modes
    modes = mode+['Sum']
    modes_legend = {mode_number: color for mode_number, color in zip(modes, colors)}
    handles = [plt.Rectangle((0, 0), 1, 1, color=modes_legend[mode_number]) for mode_number in modes]
    labels = modes
    fig.legend(handles, labels, title='Modes', loc=(0.88, 0.25), bbox_to_anchor=(1, 0.4))  
    
    # Set y-axis label for appropriate subplots
    for i in range(len(axes)):
        if i % 2 == 0:
            axes[i].set_ylabel('Depth (m)')
     
    return fig, axes
  
def Plot_η_depth_variablity_compariation(filtered_temp_dict1,
                                         filtered_temp_dict2,
                                         parameter_name,depth_round_ratio=10):
    #plot set up
    cols = len(parameter_name)
    rows= 1
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5),sharey=True)
    axes = axes.flatten()
    figure_order = [list(string.ascii_lowercase)[i % 26] for i in range(len(parameter_name))]
    #calculate the sum
    for parameter_order, parameter_name in enumerate(parameter_name):
        # Filter dictionary based on parameter name and mode number
        parameter_dict = {key: variable for key, variable in filtered_temp_dict1.items() 
                              if key[1] == parameter_name}
        parameter_df = Processing.Transfer_to_plot_df(parameter_dict,depth_round_ratio=depth_round_ratio)
        # Calculate median and IQR for each depth level    
        grouped_depth = parameter_df.groupby('depths_round')['values']
        # Calculate median and IQR for each depth level
        median_all_modes = grouped_depth.median()
        upper_error      = grouped_depth.quantile(0.75) - grouped_depth.median()
        lower_error      = grouped_depth.median() - grouped_depth.quantile(0.25)
        #plot
        axes[parameter_order].errorbar(median_all_modes,
                                       median_all_modes.index,
                                       xerr=[lower_error,upper_error], color='black',
                                       capsize=3, label = 'Disp')
        
        #Temp parameters
        Temp_parameter_df_plot_median = filtered_temp_dict2.groupby('depth_round')[parameter_name].median()
        Temp_parameter_df_plot_upper_error    = filtered_temp_dict2.groupby('depth_round')[parameter_name].quantile(0.75)-Temp_parameter_df_plot_median
        Temp_parameter_df_plot_lower_error    = Temp_parameter_df_plot_median - filtered_temp_dict2.groupby('depth_round')[parameter_name].quantile(0.25)
        #plot
        axes[parameter_order].errorbar(Temp_parameter_df_plot_median,
                                       Temp_parameter_df_plot_median.index,
                                       xerr=[Temp_parameter_df_plot_lower_error,Temp_parameter_df_plot_upper_error], color='darkcyan',
                                        capsize=3,label = 'Temp',alpha=0.5)
        
        axes[parameter_order].set_title(f'{parameter_name}')
        axes[parameter_order].grid(True)
   
    # Create a custom legend for the locations
    axes[2].legend(bbox_to_anchor=(1, 0.4))
    axes[0].set_ylabel('Depth(m)')
    axes[1].set_xlabel('value (K)')
        
    return fig,axes

def Plot_Model_type_distribution(df):
    # Unique values for reindexing
    unique_sites = df['Site'].unique()
    unique_model_types = df['model_type'].unique()
    unique_modes = df['Mode'].unique()

    # Table grouped by 'Site'
    site_table = df.pivot_table(index='Site', 
                                columns='model_type', 
                                aggfunc='size', 
                                fill_value=0)
    site_table = site_table.reindex(index=unique_sites, columns=unique_model_types, fill_value=0)
    site_table['Total'] = site_table.sum(axis=1)
    total_column_site = site_table.sum(axis=0)
    total_column_site.name = 'Total'
    site_table = site_table.append(total_column_site)

    # Table grouped by 'Mode'
    mode_table = df.pivot_table(index='Mode', 
                                columns='model_type', 
                                aggfunc='size', 
                                fill_value=0)
    mode_table = mode_table.reindex(index=unique_modes, columns=unique_model_types, fill_value=0)
    mode_table['Total'] = mode_table.sum(axis=1)
    total_column_mode = mode_table.sum(axis=0)
    total_column_mode.name = 'Total'
    mode_table = mode_table.append(total_column_mode)

    # Display the tables
    print("Table grouped by Site:")
    print(site_table)
    print("\nTable grouped by Mode:")
    print(mode_table)




def Plot_map(bathmetry_file,M2tide_SSH_file,site_names,lat_list,lon_list):
    
    #Read the tide SSH
    M2tide_SSH  = xr.open_dataset(M2tide_SSH_file)
    # Define the latitude and longitude ranges (NWS and Timor sea)
    min_lat, max_lat = -22, -8
    min_lon, max_lon = 110, 135
    # Extract variables
    X_SSH = M2tide_SSH['longitude']
    Y_SSH = M2tide_SSH['latitude']
    # M2re_SSH = M2tide_SSH['M2re']
    M2re_SSH = np.sqrt(np.power(M2tide_SSH['M2re'],2)+np.power(M2tide_SSH['M2im'],2))
    # Find the indices corresponding to the latitude and longitude ranges
    idx_X_SSH = np.where((X_SSH>min_lon) & (X_SSH<max_lon))
    X_SSH = X_SSH[(X_SSH>min_lon) & (X_SSH<max_lon)]
    idx_Y_SSH = np.where((Y_SSH>min_lat) & (Y_SSH<max_lat))
    Y_SSH = Y_SSH[(Y_SSH>min_lat) & (Y_SSH<max_lat)]
    M2re = M2re_SSH.sel(longitude = X_SSH,latitude = Y_SSH)

    #Read the bathmetry
    # Load bathymetric data
    bathmetry = xr.open_dataset(bathmetry_file)
    # Extract variables
    X_bath = bathmetry['lon']
    Y_bath = bathmetry['lat']
    idx_X_bath = np.where((X_bath>min_lon) & (X_bath<max_lon))
    X_bath = X_bath[(X_bath>min_lon) & (X_bath<max_lon)]
    idx_Y_bath = np.where((Y_bath>min_lat) & (Y_bath<max_lat))
    Y_bath = Y_bath[(Y_bath>min_lat) & (Y_bath<max_lat)]
    topo = bathmetry['elevation'].sel(lon = X_bath,lat= Y_bath)


    # Location data
    locations = {site: {'latitude': lat, 'longitude': lon} for site, lat, lon in zip(site_names, lat_list, lon_list)}
    #Plot the map
    # Create a map using PlateCarree projection
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},figsize=(25, 12))
    # Specify the contour levels you want to display
    contour_levels = [-500,-200]
    # Plot contour lines for specified levels
    contour_plot = plt.contour(X_bath, Y_bath, topo, levels=contour_levels,colors='gray',linestyles='solid')
    # Add contour labels
    # plt.clabel(contour_plot, inline=True, fontsize=10, fmt='%1.0f',colors = 'black')    
     
    # Plot the contour plot on top of the map
    contour_plot = ax.contourf(X_SSH, Y_SSH, M2re*100, cmap='cmo.amp')
    # Add colorbar inside the figure
    cax = ax.inset_axes([0.65, 0.08, 0.3, 0.04])  # [left, bottom, width, height]
    cbar = plt.colorbar(contour_plot, cax=cax,orientation='horizontal')
    cbar.set_label('M2 Amplitude (cm)',fontsize=25)
    
    # Set the extent of the map
    ax.set_extent([min_lon, max_lon, min_lat, max_lat])
    # Add Natural Earth land and ocean features
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor=cfeature.COLORS['land'])
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor=cfeature.COLORS['water'])
    ax.add_feature(land, zorder=1)
    ax.add_feature(ocean, zorder=0)
    # Add coastlines and gridlines
    ax.coastlines()
    # Use Cartopy's gridlines to control tick sizes
    gridlines = ax.gridlines(draw_labels=True)
    # Add markers and labels for specified locations
    for location, data in locations.items():
        ax.plot(data['longitude'], data['latitude'], 'ro', markersize=10, transform=ccrs.PlateCarree())
        ax.text(data['longitude'] + 0.5, data['latitude'], location, transform=ccrs.PlateCarree(),fontsize=25,fontweight='bold')
    
    ax.set_title('')
    ax.grid(True)
    # Adjust layout
    fig.tight_layout()
    # Show the map
    plt.show()
    return fig, ax


def Plot_predictions_with_RMSE(y_prediction, 
                               x_obs, y_obs, x_true, y_true, 
                               amplitude_ylim=(-100, 100), figsize=(20, 6)):
    fig, ax1 = plt.subplots(figsize=figsize)

    # redefine
    x_obs_rel = (x_obs - x_obs[-1])   # in days, so last obs is at 0
    x_true_rel = (x_true - x_obs[-1])   # starts at 0
    x_prediction_rel = (y_prediction.time - x_obs[-1]) # spans both

    # Primary y-axis: amplitude
    ax1.axvline(x=0, color='b',)  # Transition line
    ax1.plot(x_obs_rel, y_obs, '.', color='orange', label='obs')
    Plot_prediction_list(x_prediction_rel, y_prediction.Amplitude[1:], 
                         color='0.5',alpha=0.1,label='GP samples', ax=ax1)  
    ax1.plot(x_true_rel, y_true,'-.', color='orange', label='true')   
    ax1.plot(x_prediction_rel, y_prediction.Amplitude[0],color='k',lw=0.7, label='GP mean')
    ax1.set_ylabel('Amplitude (m)', color='orange')
    ax1.set_xlabel('Time (days)')
    ax1.xaxis.set_major_locator(plt.MaxNLocator(nbins=13))  # Add this
    ax1.tick_params(axis='y', labelcolor='orange')
    #ax1.set_xlim(-1,3)  # Set x-axis limits to cover both obs and true
    ax1.set_ylim(*amplitude_ylim)

    # Title
    ax1.set_title('{} samples of predictions for obs from {}'.format(
        len(y_prediction.sample),y_prediction.attrs['obs_start']))
    # compute skill - RMSE
    SE= Skill.Cal_skill(y_true, x_obs, y_prediction,skill_type='SE')
    # Secondary y-axis: RMSE
    ax2 = ax1.twinx()
    ax2.scatter(x_true_rel, np.sqrt(np.mean(SE,axis=0)), color='tab:red', label='RMSE',alpha=0.1,s=10)
    ax2.set_ylabel('RMSE (m)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(0,amplitude_ylim[1])

    # Combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2,loc='upper left')
    plt.tight_layout()
    return fig, ax1, ax2

def Plot_best_prediction_sample(GP_predict_dataset, 
                                x_obs, y_obs, x_true, y_true,
                                amplitude_ylim=(-100, 100), figsize=(20, 6)):
    # Compute squared error
    y_prediction = GP_predict_dataset['Amplitude'].where(GP_predict_dataset['time'] > x_obs[-1],drop=True)
    # exclude the mean, i.e. the first sample
    SE = (y_prediction - y_true.values)**2
    RMSE = np.sqrt(SE.mean(dim='time'))  # shape: (sample,)
    best_sample_idx = RMSE.argmin().item()
    best_rmse = np.sqrt(SE[best_sample_idx])
    print(f'Best sample index: {best_sample_idx}')
    # plot
    fig, ax = plt.subplots(figsize=figsize)
    # Time references
    x_obs_rel = x_obs - x_obs[-1]
    x_true_rel = x_true - x_obs[-1]
    x_prediction_rel = y_prediction.time - x_obs[-1]
    # Plot obs
    ax.axvline(x=0, color='b')  # transition line
    ax.plot(x_obs_rel, y_obs, '.', color='orange', label='obs')
    # Plot best sample
    best_sample = y_prediction.sel(sample=best_sample_idx)
    ax.plot(x_prediction_rel, best_sample, color='k', lw=1.5, label='Best prediction')
    # Plot true signal
    ax.plot(x_true_rel, y_true, '-.', color='orange', lw=1, label='true')
    ax.set_ylabel('Amplitude (m)s', color='orange')
    ax.set_xlabel('Time (days)')
    ax.set_ylim(*amplitude_ylim)
    ax.set_title(f'Best GP sample (#{best_sample_idx}) — obs from {GP_predict_dataset.attrs.get("obs_start", "unknown")}')
    # Right axis for RMSE
    ax2 = ax.twinx()
    ax2.scatter(x_true_rel, best_rmse, color='red', label=f'RMSE',alpha=0.25,s=10)
    ax2.set_ylabel('RMSE (m)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0,amplitude_ylim[1])
    # Combine legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper right')
    return fig, ax

def Compute_MSE(SE_list):
    all_samples = [sample for time_series in SE_list for sample in time_series]
    return np.mean(np.stack(all_samples),axis=0)

def Plot_RMSE(SE_list, x_true, window_length=1,alpha=0.1,
              label_prefix="", ax=None, line_color=None,):
    
    # Convert to DataArray
    RMSE_list = np.sqrt(np.mean(SE_list,axis=1))
    MSE_list = Compute_MSE(SE_list)
    skill_da = xr.DataArray(RMSE_list.copy(),dims=['samples', 'time'],
                            coords={'samples': range(len(RMSE_list)),'time': x_true})
    #Process time array
    time_array = skill_da.time - skill_da.time[0]  # Convert to relative time in days
    dt_days = (time_array[1] - time_array[0])
    window_size = max(int(window_length / dt_days), 1)
    skill_smooth = skill_da.rolling(time=window_size, center=True).mean()
    #skill_smooth = skill_smooth.dropna(dim='time')
    # Smooth the mean MSE across samples for the main line
    MSE_da = xr.DataArray(MSE_list.copy(), dims=['time'], coords={'time': x_true})
    smoothed_MSE = MSE_da.rolling(time=window_size, center=True).mean()
    #smoothed_MSE = smoothed_MSE.dropna(dim='time')
    ax.plot(time_array,np.sqrt(smoothed_MSE), label=f'{label_prefix,len(RMSE_list)}',color=line_color, lw=2)
    #plot
    for i in skill_smooth:
        ax.plot(time_array,i,color=line_color,alpha=alpha)
    # ax.set_xlabel("Time (days)")
    ax.set_ylabel('RMSE (m)')
    ax.legend(loc='lower right')
    ax.grid(True)
    return MSE_list

def Plot_skill(skill_list, x_true, site, medium_T, mode=1,window_length=1,skill_type='RMSE',
              label_prefix="", ax=None, line_color=None,line_style='-',raw_plot=True):
    # Convert to DataArray
    if len(skill_list) == len(x_true): #MSE_skill
        skill_da = xr.DataArray(np.sqrt(skill_list.copy()),dims=['time'],
                                coords={'time': x_true})
    else:
        skill_da = xr.DataArray(skill_list.copy(),dims=['samples', 'time'],
                                coords={'samples': range(len(skill_list)),
                                        'time': x_true})
        ## mean over samples
        skill_da = skill_da.mean(dim='samples')
    #Process time array
    time_array = skill_da.time - skill_da.time[0]  # Convert to relative time in days
    dt_days = (time_array[1] - time_array[0])#.astype('timedelta64[D]').astype('float')  # Convert to days
    window_size = max(int(window_length / dt_days), 1)
    skill_smooth = skill_da.rolling(time=window_size, center=True).mean() 
    # skill_smooth = skill_smooth.dropna(dim='time')   
    #Read the parameters and compute the timescale
    if mode is None:
        param = medium_T.loc[(site,)]
    else:
        param = medium_T.loc[(site, mode)]
    if len(param) == 6:  # M1P2
        NPL_var = param['η_c']**2 + param['η_M2']**2 + param['η_S2']**2
        T_D2 = (param['T_M2'] * param['η_M2'] + param['T_S2'] * param['η_S2']) / \
               (param['η_M2'] + param['η_S2'])
    elif len(param) == 4: # M1P1
        if 'η_c' in param: #M1P1
            NPL_var = param['η_c']**2 + param['η_D2']**2 
            T_D2 = param['T_D2']
        else: #P2
            NPL_var = param['η_M2']**2 + param['η_S2']**2
            T_D2 = (param['T_M2'] * param['η_M2'] + param['T_S2'] * param['η_S2']) / \
               (param['η_M2'] + param['η_S2'])
    elif len(param) == 2:
        NPL_var = param['η_D2']**2 
        T_D2 = param['T_D2']
    #Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))
    else:
        fig = ax.figure
    if line_color is None:
        line_color = next(ax._get_lines.prop_cycler)['color']
    ##plot skill
    if raw_plot:
        ax.plot(time_array, skill_da, label=f'{label_prefix,skill_type}', 
                marker='.', color=line_color, alpha=0.05)
    ax.plot(time_array, skill_smooth, label=f'{label_prefix,skill_type} (smooth)', 
            color=line_color, lw=2, linestyle=line_style)
    # Plot timescale line and label if new or changed
    existing_TD2_lines = [line for line in ax.lines if line.get_linestyle() == '--']
    existing_TD2_positions = [line.get_xdata()[0] for line in existing_TD2_lines]  # Extract x positions
    ylim_top = max(skill_smooth.max(), ax.get_ylim()[1]) 
    if not np.isclose(T_D2, existing_TD2_positions, atol=1e-6).any():  # If T_D2 not already plotted
        ax.axvline(x=T_D2, color='k', linestyle='--', lw=1)
        ax.text(T_D2, ylim_top * 0.98, f'{label_prefix} T_D2', color='k',
                ha='right', va='top')
    # if skill_type == 'RMSE':
    #     eta_val = np.sqrt(NPL_var)
    #     existing_hlines = [line for line in ax.lines if line.get_linestyle() == ':']
    #     if not existing_hlines:
    #         ax.axhline(y=eta_val, color='k', linestyle=':', lw=1)
    #         ax.text(ax.get_xlim()[1]*0.95, eta_val, f'{label_prefix} η', color='k',
    #                 ha='right', va='bottom')
    # ax.set_xlabel("Time (days)")
    ylim_top = max(skill_smooth.max(), ax.get_ylim()[1]) 
    ax.set_ylim(0, ylim_top*1.01)  # Ensure y-axis starts at 0
    ax.legend(loc='lower right')
    ax.grid(True)
    return fig, ax


def Plot_log_fit(skill_list, x_true, threshold_ratio=0.10,label_prefix="", ax=None,
                 line_color=None):
    """
    Plot original RMSE and fitted logarithmic curve with threshold marker.
    """
    skill_fit, t_thresh, (a,b,c) = Processing.Fit_logarithmic_growth(skill_list,x_true,
                                                                    threshold_ratio)
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))
    else:
        fig = ax.figure
    # ax.plot(time_array, RMSE_array, 'o-', label='RMSE (raw)', alpha=0.4)
    time_array = x_true - x_true[0]
    ax.plot(time_array, skill_fit, '--',linewidth=3, color=line_color,alpha=0.5)
    ax.axvline(t_thresh, linestyle=':', color=line_color,
               label=f"t_threshold_{label_prefix} ≈ {t_thresh:.2f} days")
    ax.text(t_thresh + 0.1, ax.get_ylim()[1]*0.98, f'{label_prefix} t_thresh', 
        color=line_color, ha='left', va='top')
    # ax.set_xlabel("Time (days)")
    # ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(True)

    return fig, ax