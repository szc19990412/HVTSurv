import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='UCEC/HVTSurv.yaml',type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = make_parse()

    #---->file address
    log_path = '/data112/shaozc/HVTSurv/logs/'
    log_name = Path(args.config).parent 
    version_name = Path(args.config).name[:-5]
    log_path = Path(log_path) / log_name / version_name
    all_risk_scores_list = list(log_path.glob('*/all_risk_scores.npz'))
    all_censorships_list = list(log_path.glob('*/all_censorships.npz'))
    all_event_times_list = list(log_path.glob('*/all_event_times.npz'))

    #---->Record the low+high of each fold
    Low_risk_all = pd.DataFrame(columns=['risk', 'time', 'event'])
    High_risk_all = pd.DataFrame(columns=['risk', 'time', 'event'])

    for i in range(4):
        result_dir = all_risk_scores_list[i]
        all_risk_scores = np.load(result_dir,allow_pickle=True)['arr_0'].tolist()
        result_dir = all_censorships_list[i]
        all_censorships = np.load(result_dir,allow_pickle=True)['arr_0'].tolist()
        result_dir = all_event_times_list[i]
        all_event_times = np.load(result_dir,allow_pickle=True)['arr_0'].tolist()

        predictions = pd.DataFrame({'risk':all_risk_scores, 'time':all_event_times, 'event':list(1-np.array(all_censorships))})

        #---->find censored data
        patient_df = predictions.drop_duplicates(['time']).copy()
        patient_df.reset_index(drop=True, inplace=True)
        df_censor = patient_df.sort_values('time').reset_index(drop=True)
        censor_index = list(df_censor.loc[df_censor.event==0].index)

        #---->KM curve drawing
        with plt.style.context(['science','ieee','high-contrast']):
            plt.figure()
            kmf = KaplanMeierFitter()
            kmf.fit(durations=predictions['time'].values,
                    event_observed=predictions['event'].values)
            x = kmf.survival_function_.index.values
            y = kmf.survival_function_

            #---->Only censored data draw a plus sign
            plt.plot(x, y, '+-', markevery=censor_index)
            plt.ylim(0, 1.1)
            plt.xlim(None, 300)
            plt.title('')
            plt.xlabel('Time (months)')
            plt.ylabel('Survival probability')
            plt.savefig(log_path/ f'fold{i}' / 'KM.png', bbox_inches = 'tight') 

            #---->According to the risk index into high and low
            median_prob = np.median(all_risk_scores)
            risk_groups = {'Low-risk': np.where(all_risk_scores<median_prob),
                        'High-risk': np.where(all_risk_scores>=median_prob)}
            
            #---->Plot the curve of p-value
            PLOT_SIZE = (4, 2.75)
            PLOT_SIZE = (3.75, 2.5)
            default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            fig = plt.figure(figsize=PLOT_SIZE)
            ax = fig.add_subplot(1, 1, 1)
            group_colors = {
            'Low-risk': default_colors[-2],
            'High-risk': default_colors[0]}
            
            #---->Put low-risk and high-risk into risk_groups
            Low_risk = predictions.loc[risk_groups['Low-risk'][0].tolist()]
            High_risk = predictions.loc[risk_groups['High-risk'][0].tolist()]
            Low_risk.reset_index(drop=True, inplace=True)
            High_risk.reset_index(drop=True, inplace=True)

            #---->Save the low+high of each fold
            Low_risk_all = Low_risk_all.append(Low_risk)
            High_risk_all = High_risk_all.append(High_risk)

            #---->find censored data
            patient_df_low = Low_risk.drop_duplicates(['time']).copy()
            patient_df_low.reset_index(drop=True, inplace=True)
            df_censor_low = patient_df_low.sort_values('time').reset_index(drop=True)
            df_low_max = df_censor_low['time'].tolist()[-1] #the longest surviving
            censor_index_low = list(df_censor_low.loc[df_censor_low.event==0].index)

            #---->Draw the KM curves of low-risk and high-risk
            kmf_low = KaplanMeierFitter()
            kmf_low.fit(durations=Low_risk['time'].values,
                event_observed=Low_risk['event'].values)
            
            x = kmf_low.survival_function_.index.values
            y = kmf_low.survival_function_
            ax.plot(x, y, '+-', color=group_colors['Low-risk'], label='Low-risk', markevery=censor_index_low)

            #---->find censored data
            patient_df_high = High_risk.drop_duplicates(['time']).copy()
            patient_df_high.reset_index(drop=True, inplace=True)
            df_censor_high = patient_df_high.sort_values('time').reset_index(drop=True)
            df_high_max = df_censor_high['time'].tolist()[-1] #the longest surviving
            censor_index_high = list(df_censor_high.loc[df_censor_high.event==0].index)

            kmf_high = KaplanMeierFitter()
            kmf_high.fit(durations=High_risk['time'].values,
                event_observed=High_risk['event'].values)
            x = kmf_high.survival_function_.index.values
            y = kmf_high.survival_function_
            ax.plot(x, y, '+-', color=group_colors['High-risk'], label='High-risk', markevery=censor_index_high)
            ax.set_ylim(0, 1.1)
            time_max = np.max([int(df_high_max), int(df_low_max)])+1
            ax.set_xlim(None, time_max)

            #---->Adjustment map (Color + Line style)
            legend_elements = []
            legend_elements.append(matplotlib.patches.Patch(
                facecolor=group_colors['High-risk'], edgecolor=group_colors['High-risk'],
                linewidth=3, label='High-risk'))
            legend_elements.append(matplotlib.patches.Patch(
                    facecolor=group_colors['Low-risk'], edgecolor=group_colors['Low-risk'],
                    linewidth=3, label='Low-risk'))

            # markers, line_styles = ['|', ''], ['-', '--']
            # legend_elements.append(matplotlib.lines.Line2D(
            #         [0], [0], color='k', marker=markers[0], ls=line_styles[0], label='Kaplan-Meier'))

            ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_title('')
            ax.set_xlabel('Time (months)')
            ax.set_ylabel('Survival probability')

            #---->Calculate p-value
            lr_test = logrank_test(durations_A=kmf_low.durations,
                                durations_B=kmf_high.durations,
                                event_observed_A=kmf_low.event_observed,
                                event_observed_B=kmf_high.event_observed)
            ax.text(152/300*time_max, 0.75, 'p', style='italic')
            base, exp = f'{lr_test.p_value:.1e}'.split('e')
            txt = r'= ${0:}^{{{1:}}}$'.format(base, exp)
            ax.text(164/300*time_max, 0.75, txt)

            #---->save fig
            ax.figure.savefig(log_path/ f'fold{i}' / 'Log-rank.png', bbox_inches = 'tight', dpi=300) 

    #----> all the data
    all_risk_scores = []
    all_censorships = []
    all_event_times = []

    for result_dir in all_risk_scores_list:
        all_risk_scores = all_risk_scores + np.load(result_dir,allow_pickle=True)['arr_0'].tolist()
    for result_dir in all_censorships_list:
        all_censorships = all_censorships + np.load(result_dir,allow_pickle=True)['arr_0'].tolist()
    for result_dir in all_event_times_list:
        all_event_times = all_event_times + np.load(result_dir,allow_pickle=True)['arr_0'].tolist()

    #---->
    predictions = pd.DataFrame({'risk':all_risk_scores, 'time':all_event_times, 'event':list(1-np.array(all_censorships))})

    #---->
    patient_df = predictions.drop_duplicates(['time']).copy()
    patient_df.reset_index(drop=True, inplace=True)
    df_censor = patient_df.sort_values('time').reset_index(drop=True)
    censor_index = list(df_censor.loc[df_censor.event==0].index)

    #---->
    with plt.style.context(['science','ieee']):
        plt.figure()
        kmf = KaplanMeierFitter()
        kmf.fit(durations=predictions['time'].values,
                event_observed=predictions['event'].values)
        kmf.fit(durations=predictions['time'].values,
                    event_observed=predictions['event'].values)
        x = kmf.survival_function_.index.values
        y = kmf.survival_function_
        plt.plot(x, y, '+-', markevery=censor_index)
        plt.ylim(0, 1.1)
        plt.xlim(None, 300)
        plt.title('')
        plt.xlabel('Time (months)')
        plt.ylabel('Survival probability')
        plt.savefig(log_path / 'KM.png', bbox_inches = 'tight') #进一步保存到fold底下

        # #---->
        # median_prob = np.median(all_risk_scores)
        # risk_groups = {'Low-risk': np.where(all_risk_scores<median_prob),
        #             'High-risk': np.where(all_risk_scores>=median_prob)}
        
        #---->
        PLOT_SIZE = (4, 2.75)
        PLOT_SIZE = (3.75, 2.5)
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig = plt.figure(figsize=PLOT_SIZE)
        ax = fig.add_subplot(1, 1, 1)
        group_colors = {
        'Low-risk': 'b',
        'High-risk': 'r'}
        
        #---->
        # Low_risk = predictions.loc[risk_groups['Low-risk'][0].tolist()]
        # High_risk = predictions.loc[risk_groups['High-risk'][0].tolist()]
        Low_risk = Low_risk_all
        High_risk = High_risk_all
        Low_risk.reset_index(drop=True, inplace=True)
        High_risk.reset_index(drop=True, inplace=True)

        #---->
        patient_df_low = Low_risk.drop_duplicates(['time']).copy()
        patient_df_low.reset_index(drop=True, inplace=True)
        df_censor_low = patient_df_low.sort_values('time').reset_index(drop=True)
        df_low_max = df_censor_low['time'].tolist()[-1] #
        censor_index_low = list(df_censor_low.loc[df_censor_low.event==0].index)

        #---->
        kmf_low = KaplanMeierFitter()
        kmf_low.fit(durations=Low_risk['time'].values,
            event_observed=Low_risk['event'].values)
        
        x = kmf_low.survival_function_.index.values
        y = kmf_low.survival_function_
        ax.plot(x, y, '+-', color=group_colors['Low-risk'], label='Low-risk', markevery=censor_index_low)

        #---->
        patient_df_high = High_risk.drop_duplicates(['time']).copy()
        patient_df_high.reset_index(drop=True, inplace=True)
        df_censor_high = patient_df_high.sort_values('time').reset_index(drop=True)
        df_high_max = df_censor_high['time'].tolist()[-1] 
        censor_index_high = list(df_censor_high.loc[df_censor_high.event==0].index)

        kmf_high = KaplanMeierFitter()
        kmf_high.fit(durations=High_risk['time'].values,
            event_observed=High_risk['event'].values)
        x = kmf_high.survival_function_.index.values
        y = kmf_high.survival_function_
        ax.plot(x, y, '+-', color=group_colors['High-risk'], label='High-risk', markevery=censor_index_high)
        ax.set_ylim(0, 1.1)
        time_max = np.max([int(df_high_max), int(df_low_max)])+1
        ax.set_xlim(None, time_max)

        #---->
        legend_elements = []
        legend_elements.append(matplotlib.patches.Patch(
            facecolor=group_colors['High-risk'], edgecolor=group_colors['High-risk'],
            linewidth=3, label='High-risk'))
        legend_elements.append(matplotlib.patches.Patch(
                facecolor=group_colors['Low-risk'], edgecolor=group_colors['Low-risk'],
                linewidth=3, label='Low-risk'))

        markers, line_styles = ['|', ''], ['-', '--']
        # legend_elements.append(matplotlib.lines.Line2D(
        #         [0], [0], color='k', marker=markers[0], ls=line_styles[0], label='Kaplan-Meier'))

        ax.legend(handles=legend_elements, loc='lower left',frameon=True ,facecolor='#ffffff')#, bbox_to_anchor=(1, 0.5))
        # ax.set_title(f'{log_name}', fontsize = 11)
        ax.set_title('CO\&RE', fontsize = 11)
        ax.set_xlabel('Time (months)', fontsize = 10)
        ax.set_ylabel('Survival probability', fontsize = 10)

        #---->
        lr_test = logrank_test(durations_A=kmf_low.durations,
                            durations_B=kmf_high.durations,
                            event_observed_A=kmf_low.event_observed,
                            event_observed_B=kmf_high.event_observed)
        ax.text(132/300*time_max, 0.95, 'P-Value', style='italic', fontsize = 11)
        base, exp = f'{lr_test.p_value:.1e}'.split('e')
        txt = r'= ${0:}^{{{1:}}}$'.format(base, exp)
        ax.text(194/300*time_max, 0.95, txt, fontsize = 11)
        ax.text(254/300*time_max, 0.95, '(*)', fontsize = 11)

        #---->
        ax.figure.savefig(log_path / 'Log-rank.png', bbox_inches = 'tight', dpi=300) 