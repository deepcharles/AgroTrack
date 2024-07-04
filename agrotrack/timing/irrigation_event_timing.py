import os
import xarray as xr
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
import ruptures as rpt 

def irrigation_event_timing(st_data, st_info, year, df_binary, df_insitu_irrigation, model = 'l2', penalty = 1, min_seg_size = 3, segmentation_method = 'mean',mean_deltalst_quantile_cutoff = 0.5, add_plot = True): 
    
    from agrotrack import irrigation_season_timing_point
    # Segmentation after identifying the break points
    def irrigation_segment_detector(mean_deltalst_quantile_cutoff,final_bps,diff_lst_ir_season,segmentation_method):
        # calculatiing mean delta lst for each segment

        bp_start = 0
        diff_lst_segs = []
        for bp in final_bps:
            diff_lst_seg = diff_lst_ir_season.isel(time = slice(bp_start,bp)).mean(dim = 'time')
            diff_lst_segs.append(diff_lst_seg)
            bp_start = bp
        diff_lst_seg_mean = xr.concat(diff_lst_segs,'time')
        # if add_plot == True:
        #     diff_lst_seg_mean.plot(x = 'time',figsize = [18,10],linestyle='dashed',color='g', linewidth=2, marker='o',markersize=10)
        seg_irr_flag = (diff_lst_seg_mean<diff_lst_seg_mean.quantile(mean_deltalst_quantile_cutoff)).astype(int)

        # calculating mean gradient within each segment
        datetime = diff_lst_ir_season["time"]
        diff_lst_ir_season["time"] = diff_lst_ir_season["time"].dt.dayofyear
        bp_start = 0
        lst_seg_grads = []
        for bp in final_bps:
            lst_seg_grad = diff_lst_ir_season.isel(time = slice(bp_start,bp)).differentiate("time"). mean(dim = 'time')
            lst_seg_grads.append(lst_seg_grad)
            bp_start = bp
        lst_seg_grad_mean = xr.concat(lst_seg_grads,'time')
        # if add_plot == True:
        #     lst_seg_grad_mean.plot(x = 'time',figsize = [18,10],linestyle='dashed',color='r', linewidth=2, marker='o',markersize=10)
        # seg_irr_flag = (diff_lst_seg_mean<diff_lst_seg_mean.quantile(mean_deltalst_quantile_cutoff)).astype(int)
        diff_lst_ir_season["time"] = datetime

        delta_lst_seg_mean_grad_combined = xr.merge([diff_lst_seg_mean.rename("segments mean"),lst_seg_grad_mean.rename("segments mean gradient")])

        seg_mean_grad = delta_lst_seg_mean_grad_combined.to_dataframe()
        # print('number of samples: ', len(seg_mean_grad))                               
        # filling the gap between the break points (which segment is irrigated and wich one is not)
        if segmentation_method == 'mean':
            irr_segs = np.zeros_like(diff_lst_ir_season)
            bp_start = 0
            for i,bp in enumerate(final_bps):
                irr_segs[bp_start: bp]= seg_irr_flag[i]
                bp_start = bp

            irr_segs
            irr_seg = xr.zeros_like(diff_lst_ir_season)
            irr_seg = irr_seg.rename("detected segments")
            irr_seg.values = irr_segs
            df_detected_seg = irr_seg.to_dataframe()  
        elif segmentation_method == 'trend':
            irr_segs = np.zeros_like(diff_lst_ir_season)
            bp_start = 0
            prev_seg_mean = 0
            for i,bp in enumerate(final_bps):
                if (diff_lst_seg_mean[i]<prev_seg_mean):
                    irr_segs[bp_start: bp]= 1
                elif (diff_lst_seg_mean[i]-prev_seg_mean)<diff_lst_seg_mean.std(dim = 'time') and diff_lst_seg_mean[i]<diff_lst_seg_mean.quantile(mean_deltalst_quantile_cutoff): # if the segment mean has increased but not significantly it can be and irrigation episode
                    irr_segs[bp_start: bp]= 1
                bp_start = bp
                prev_seg_mean =diff_lst_seg_mean[i]

            irr_segs
            irr_seg = xr.zeros_like(diff_lst_ir_season)
            irr_seg = irr_seg.rename("detected segments")
            irr_seg.values = irr_segs
            df_detected_seg = irr_seg.to_dataframe()
        elif segmentation_method == 'kmean': 
            from sklearn.cluster import KMeans
            features = seg_mean_grad[["segments mean","segments mean gradient"]].values
            kmeans= KMeans(n_clusters=3,random_state=0).fit(features)
            kmean_flags = kmeans.labels_
            centeroid = kmeans.cluster_centers_

            # we are working under the assumption that the irrigated segments having lower mean Delta LST
            irrigated_idx = np.argmin(centeroid[:,0])
            kmean_flags = (kmean_flags == irrigated_idx).astype(int)

            irr_segs = np.zeros_like(diff_lst_ir_season)
            bp_start = 0
            for i,bp in enumerate(final_bps):
                irr_segs[bp_start: bp]= kmean_flags[i]
                bp_start = bp

            irr_segs
            irr_seg = xr.zeros_like(diff_lst_ir_season)
            irr_seg = irr_seg.rename("detected segments")
            irr_seg.values = irr_segs

            df_detected_seg = irr_seg.to_dataframe()
        return df_detected_seg
    
    # trimming invalid data from start and end of the df
    def trim_invalid_data(df):
        valid = (df.notna() & (df != 0))
        valid_st = valid.any(axis = 1).idxmax()
        valid_en = valid.any(axis =1)[::-1].idxmax()
        trimmed_df = df.loc[valid_st:valid_en]
        return trimmed_df
    
    
    # detection of the break points within the irrigation season    
    stid = st_info['name']
    bp0,bp1 = irrigation_season_timing_point(st_data, st_info, model = 'l2', add_plot = False)
    diff_lst_ir_season = st_data.delta_lst_nonan_st.sel(time=slice(bp0,bp1)) # slicing the time series for the irrigation period
    algo = rpt.KernelCPD(kernel="linear", min_size=min_seg_size).fit(diff_lst_ir_season.values)
    my_bkps = algo.predict(pen=penalty)
    
    if add_plot == True:
        from agrotrack import display
        display(diff_lst_ir_season.values, my_bkps,show_piecewise_linear=True)
        
    diff_lst_ir_arr = diff_lst_ir_season.values
    # diff_sm_ir_arr = diff_sm_ir_season.values

    date_ir = diff_lst_ir_season['time'].values
    final_bps = []
                                   
    for bp in my_bkps[:-1]:
        if bp+1<len(diff_lst_ir_arr)-1: 
            final_bps.append(bp)

    bps = date_ir[final_bps]
    bps_diff_lst = diff_lst_ir_arr[final_bps]
    lst_ir_ir_season = st_data.lst_ir_st.sel(time=slice(bp0,bp1))
    lst_ir_arr = lst_ir_ir_season.values
    final_bps.append(len(lst_ir_arr))
    
    # identifying the the irrigated segments between the break points
    df_detected_seg = irrigation_segment_detector(mean_deltalst_quantile_cutoff,final_bps,diff_lst_ir_season,segmentation_method)
    
    # finding the start and end date of each irrigation episode
    if add_plot == True:
        diff_lst_ir_season.plot(x = 'time',figsize = (15,8))
        label0 = bp0.dt.strftime("%b %d, %Y")
        plt.axvline(x=bp0.values, color='g', label=f'start: {label0}',linestyle='--')
        label1 = bp1.dt.strftime("%b %d, %Y")
        plt.axvline(x=bp1.values, color='r', label=f'end: {label1}',linestyle='--')
        plt.axvspan(bp0.values, bp1.values, color='g', alpha=0.1)
        plt.plot(bps,bps_diff_lst,'ro',label = 'potential irrigation application based on diff LST break points')
        plt.legend(loc= 'lower center', ncol =2)
        plt.title(f'station code: {stid}')

    df_bps = pd.DataFrame(np.zeros([366,1]),columns = ['Break points'])
    df_bps['date'] = pd.date_range(start=f'{year}-01-01', periods=366,freq='D')
    df_bps.set_index('date',inplace=True)
    df_bps.loc[bps] = 1#*df_insitu_irrigation[stid[ID]].max()
    bps = pd.to_datetime(bps)

    df_insitu_irrigation.index = pd.to_datetime(df_insitu_irrigation.index)
    df_binary.index = pd.to_datetime(df_binary.index)

    df_insitu_elec  = df_insitu_irrigation[df_insitu_irrigation.index.year == int(year)] # only year 2020
    df_insitu_bin  = df_binary[df_binary.index.year == int(year)] # only year 2020
    df_bps = df_bps[df_bps.index.isin(df_insitu_bin.index)] # filtering df_bps based on df_insitu_2020
    df_combo_bin_bps = pd.concat([df_insitu_bin[stid],df_bps,df_detected_seg['detected segments']], axis=1) # ,df_bps_sm
    df_combo_bin_bps.index = df_combo_bin_bps.index.strftime('%y-%b-%d')
    df_combo_bin_bps = df_combo_bin_bps.rename(columns={f"{stid}": "observed data"})
    df_combo_bin_bps = trim_invalid_data(df_combo_bin_bps)
    
    if add_plot == True:
        mpl.rcParams["font.size"] = 22
        plt.subplots(figsize = [24,6])
        colors = ((.85, .85, 0.85), (0, 0, 0))
        cmap = LinearSegmentedColormap.from_list('Custom', colors, len(colors))
        ax = sns.heatmap(df_combo_bin_bps.fillna(0).T,cmap = cmap,cbar_kws={'label': 'irrigation (0=No, 1=Yes)'})
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([0.25,0.75])
        colorbar.set_ticklabels(['0', '1'])
        plt.title('Irrigation episodes vs detected irrigation events')
        ax.figure.tight_layout()
        
        
        png_dir = '/discover/nobackup/ejalilva/pictures/paper figures/'
        filename = f'irrigation timing binary heatmap ({stid})'
        img_filename = os.path.join(png_dir,f'{filename}.png')
        plt.savefig(img_filename, transparent = False,dpi = 150)
                                   
        df_combo_elec_bps = pd.concat([df_insitu_elec[stid],
                                       df_bps*df_insitu_irrigation[stid].quantile(.95)/2,
                                       df_detected_seg['detected segments']*df_insitu_irrigation[stid].quantile(.95)/2], axis=1)
        df_combo_elec_bps = trim_invalid_data(df_combo_elec_bps.rename(columns={f"{stid}": "observed data"}))
        df_combo_elec_bps.index = df_combo_elec_bps.index.strftime('%y-%b-%d')
        
        plt.subplots(figsize = [24,6])
        ax = sns.heatmap(df_combo_elec_bps.fillna(0).T,cmap = sns.color_palette("rocket_r", as_cmap=True),cbar_kws={'label': 'Electricity consumption (${m^3/day}$)'},vmax = df_insitu_irrigation[stid].quantile(.98))
        plt.title('Electricity consumption vs detected irrigation events')
        ax.figure.tight_layout()
        
        filename = f'irrigation timing heatmap ({stid})'
        img_filename = os.path.join(png_dir,f'{filename}.png')
        plt.savefig(img_filename, transparent = False,dpi = 150)

        df_combo_elec_bin = pd.concat([df_insitu_elec[stid],(df_insitu_irrigation[stid]*df_detected_seg['detected segments'].T).T.replace(0,np.nan)], axis=1)
        axes = df_combo_elec_bin.plot.line(figsize = [18,6],style=['c-','k*-'],linewidth = 3,fontsize = 16,grid = True)
        axes.legend(["water release", "detected irrigation episodes"]);


    # Creating the confusion matirx and other classification metrics
    irr_true = df_combo_bin_bps[f'observed data'].fillna(0)
    irr_pred = df_combo_bin_bps['detected segments'].fillna(0)
    from sklearn.metrics import confusion_matrix,accuracy_score,precision_score, recall_score, f1_score,ConfusionMatrixDisplay
    cm = confusion_matrix(irr_true,irr_pred)
    # Create a confusion matrix display object


    # Performance score
    acc = accuracy_score(irr_true,irr_pred)*100
    prec = precision_score(irr_true,irr_pred)*100
    recall = recall_score(irr_true,irr_pred)*100
    f1 = f1_score(irr_true,irr_pred)*100
    performance = [acc,prec,recall,f1]
    if add_plot == True: 
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        # Plot the confusion matrix
        disp.plot()
        plt.show()
        print('accuracy: {:.2f}%'.format(acc))
        print('precision: {:.2f}%'.format(prec))
        print('recall: {:.2f}%'.format(recall))
        print('f1_score: {:.2f}%'.format(f1))
    return performance, df_combo_bin_bps
