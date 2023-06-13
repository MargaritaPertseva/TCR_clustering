#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:39:47 2022

@author: pertsevm
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import umap
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

import os 
import argparse

#import other python files
import config_script
import model_architecture
import enc_decode


# parse the command line
if __name__ == "__main__":
    # initialize Argument Parser
    parser = argparse.ArgumentParser()
    
    # we can add any different arguments we want to parse
    parser.add_argument("--input_type", type=str) #beta, albeta, beta_VJ, albeta_VJ
    parser.add_argument("--embedding_space", type=int)
    parser.add_argument("--results_folder", type=str) #to record it

# we then read the arguments from the comand line
args = parser.parse_args()
INPUT_TYPE = args.input_type
RESULTS_FOLDER = args.results_folder
EMBEDDING_SPACE = args.embedding_space

    

#####################################################################################
################## Open test embeddings and universal plot df #######################

# 1) open the test seqs embedding file from the current model
test_emb_umap_pca = pd.read_csv(config_script.out_analysis_dir + RESULTS_FOLDER + '/' + INPUT_TYPE + '_UMAP_PCA_test_emb.csv', header = 0)

# 2) open df with selected seqs for plotting
universal_plot_df = pd.read_csv(config_script.plot_df_test, header = 0)

# 3) extract predicted model embeddings for selected seqs
plot_seqs = pd.merge(universal_plot_df,test_emb_umap_pca, on = ['CDR3a', 'CDR3b', 'species'], how = 'inner')
print ('Length should be 11880 seqs! Actual length is : ', len(plot_seqs))


#####################################################################################
########################## Calculate UMAP PCA #####################################  

def umap_pca_df(results_array, y_list, random_seed):
    
    import umap
    from sklearn.decomposition import PCA
    from sklearn.decomposition import KernelPCA
    from sklearn.preprocessing import StandardScaler
    no_of_components = 2
    pca = PCA(n_components=2, random_state=random_seed)
    u = umap.UMAP(n_components = 2, random_state=random_seed, min_dist = 0.3, n_neighbors = 15, metric = 'euclidean')
    
    
    df_umap = pd.DataFrame(u.fit_transform(results_array), columns = ['axis1','axis2'])
    df_umap['dim_reduction'] = 'umap'
    df_pca = pd.DataFrame(pca.fit_transform(results_array), columns = ['axis1','axis2'])
    df_pca['dim_reduction'] = 'pca'
    
    df_umap_pca = pd.concat([df_umap, df_pca], ignore_index = True)
    df_umap_pca['Epitope'] = y_list + y_list
    
    return df_umap_pca
    
#####################################################################################
########################## To plot latent space #####################################  
palette_22 = ['maroon','red', 'brown', 'orange', 'olive', \
            'lime', 'yellow', 'green', 'cyan', 'blue', \
            'teal', 'navy', 'pink', 'purple', 'lavender', \
            'magenta', 'peachpuff', 'beige', 'mint', 'black', \
            'white', 'grey']

palette_31 = ['#201923', '#ffffff', '#fcff5d','#7dfc00','#946aa2', \
           '#5d4c86','#0ec434','#228c68','#8ad8e8','#235b54', \
           '#29bdab','#3998f5','#37294f','#277da7','#3750db', \
           '#f22020','#991919','#ffcba5','#e68f66','#c56133', \
           '#96341c','#632819','#ffc413','#f47a22', '#2f2aa0', \
           '#b732cc','#772b9d','#f07cab','#d30b94','#edeff3', \
           '#c3a5b4']

COLOR_DICT = {'GILGFVFTL': '#201923', 'LSLRNPILV': '#ffffff', 'LTDEMIAQY': '#fcff5d', 'SSLENFRAYV': '#7dfc00', \
              'KLGGALQAK': '#946aa2', 'SPRWYFYYL': '#5d4c86', 'NLVPMVATV': '#0ec434', 'YLQPRTFLL': '#228c68', \
              'DATYQRTRALVR': '#8ad8e8', 'KAVYNFATM': '#235b54', 'LLWNGPMAV': '#29bdab', 'FQPQNGQFI': '#3998f5', \
              'TTDPSFLGRY': '#37294f', 'ASNENMETM': '#277da7', 'SSYRRPVGI': '#3750db', 'CINGVCWTV': '#f22020', \
              'ELAGIGILTV': '#991919', 'CTELKLSDY': '#ffcba5', 'KLVALGINAV': '#e68f66', 'HGIRNASFI': '#c56133', \
              'NQKLIANQF': '#96341c', 'RPRGEVRFL': '#632819', 'GLCTLVAML': '#ffc413', 'NLNCCSVPV': '#f47a22', \
              'SSPPMFRV': '#2f2aa0', 'TVYGFCLL': '#b732cc','FLCMKALLL': '#772b9d'}

def plot_umap_pca_latent(df, hue_column, color_dict, fig_title, legend_title, node_size, save, figname):
    
    # 1) set up sns style
    import seaborn as sns
    sns.set(font_scale = 2)
    sns.set_style("whitegrid")
    
    # 2) plot things
    g = sns.FacetGrid(df, col = 'dim_reduction', \
                  hue = hue_column, palette = color_dict, sharex=False, sharey=False)
    g.map(sns.scatterplot, "axis1", "axis2", alpha=1, \
          s=node_size, edgecolor = 'darkslategrey', linewidth = 0.8) #linewidth = 0; or edgecolor = None, to remove white border
    
    g.fig.set_figwidth(32)
    g.fig.set_figheight(16)
    g.set_titles(col_template= fig_title, fontweight='bold', size=25)
    
    # set axis labels
    g.axes[0,0].set_xlabel('UMAP_1')
    g.axes[0,0].set_ylabel('UMAP_2')
    g.axes[0,1].set_xlabel('PCA_1')
    g.axes[0,1].set_ylabel('PCA_2')
    #g.set_yticklabels(g.get_yticks(), size = 15)
    
    # 3) add legend and adjust the size
    g.add_legend(title = legend_title)           #(legend_data=None, title=None, label_order=None)
    plt.setp(g._legend.get_title(), fontsize=25, fontweight='bold') # change the size of legend title
    plt.setp(g._legend.get_texts(), fontsize=20) # change the size of legend text
    
    # 4) save figure
    if save:
        g.savefig(config_script.triplet_analysis_dir + figname, \
                  dpi=500, bbox_inches='tight', pad_inches=0.5)
        
    return g


########################## To plot latent space #####################################

# plot species distribution
plot_umap_pca_latent(plot_seqs, 'species', color_list = ['olive', 'cyan'], \
                     fig_title = f'Murine vs Human TCRs', legend_title = 'Species', \
                     node_size = 100, save = True, figname = f'species_{INPUT_TYPE}_{EMBEDDING_SPACE}dim.jpg')

# plot TRBV gene distribution
plot_umap_pca_latent(plot_seqs, 'Vb', default_color_list, \
                     fig_title = 'TRBV gene distribution', legend_title = 'TRBV', \
                     node_size = 100, save = True, figname = f'Vb_{INPUT_TYPE}_{EMBEDDING_SPACE}dim.jpg')



## BETA
if INPUT_TYPE == 'beta' or INPUT_TYPE == 'beta_VJ':
    # plot CDR3b gene distribution
    plot_umap_pca_latent(plot_seqs, 'Len_CDR3b', default_color_list, \
                         fig_title = 'CDR3b length distribution', legend_title = 'CDR3b length', \
                         node_size = 100, save = True, figname = f'CDR3b_len_{INPUT_TYPE}_{EMBEDDING_SPACE}dim.jpg')
    
## ALBETA
if INPUT_TYPE == 'albeta' or INPUT_TYPE == 'albeta_VJ':
    # plot CDR3 AB gene distribution
    plot_umap_pca_latent(plot_seqs, 'Len_CDR3ab', default_color_list, \
                         fig_title = 'CDR3ab length distribution', legend_title = 'CDR3ab length', \
                         node_size = 100, save = True, figname = f'CDR3ab_len_{INPUT_TYPE}_{EMBEDDING_SPACE}dim.jpg')
    
    # plot TRAV gene distribution
    plot_umap_pca_latent(plot_seqs, 'Va', default_color_list, \
                         fig_title = 'TRAV gene distribution', legend_title = 'TRAV', \
                         node_size = 100, save = True, figname = f'Va_{INPUT_TYPE}_{EMBEDDING_SPACE}dim.jpg')

    
    
########################## To plot reconstruction accuracy #####################################    

def plot_acc_bars(df, color_list, figname):
    sns.set_theme(style="whitegrid")
    sns.set(font_scale = 1.7)

    # Draw a nested barplot by species and sex
    g = sns.catplot(
        data=df, kind="bar",
        x="Data", y="Reconstr_acc", hue="dim",
        palette= color_list, alpha=1, legend = False, legend_out=True) #height=6 , ['teal', 'orange', 'red'] #errorbar="sd",
    g.despine(left=True)

    g.set(ylim=(0.5, 1))
    g.axes[0,0].set_xlabel('Similarity to the training set')
    g.axes[0,0].set_ylabel('Reconstruction accuracy')
    g.set_xticklabels(['90%', '85%', '80%', '75%', '70%'])

    g.fig.set_figwidth(12)
    g.fig.set_figheight(7.5)

    g.add_legend(title = "Latent space dimension")           #(legend_data=None, title=None, label_order=None)
    plt.setp(g._legend.get_title(), fontsize=18, fontweight='bold') # change the size of legend title
    plt.setp(g._legend.get_texts(), fontsize=18)

    g.savefig(out_analysis_dir + '/' + figname, \
                      dpi=500, bbox_inches='tight', pad_inches=0.5)
