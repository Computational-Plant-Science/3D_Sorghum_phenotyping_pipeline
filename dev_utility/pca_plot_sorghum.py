"""
Version: 1.5

Summary: PCA analysis and visualization

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

    python3 pca_plot_sorghum.py -p /home/suxing/analysis/ -f trait_sum_pca_update.xlsx



"""

#!/usr/bin/python
# Standard Libraries
import matplotlib
matplotlib.use('Agg')

from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import argparse

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.preprocessing import StandardScaler

from matplotlib.colors import hsv_to_rgb, rgb2hex

from adjustText import adjust_text

from pca import pca

    
    
    

def biplot(data, genotypes):
    
    ## perform PCA
    n = len(data.columns)
    
    pca = PCA(n_components = n)
    # defaults number of PCs to number of columns in imported data (ie number of
    # features), but can be set to any integer less than or equal to that value

    pca.fit(data)

    ## project data into PC space
    
    
    # 0,1 denote PC1 and PC2; change values for other PCs
    xvector = pca.components_[0] 
    yvector = pca.components_[1]

    xs = pca.transform(data)[:,0] 
    ys = pca.transform(data)[:,1]

    fig = plt.figure(figsize = (16,9))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 20)
    ax.set_ylabel('Principal Component 2', fontsize = 20)
    ax.set_title('PCA', fontsize = 25)
    
    #colors = plt.cm.rainbow(np.linspace(0, 1, len(xs)))
    
    colors = plt.cm.Spectral(np.linspace(0, 1, len(xs)))
    
    ## visualize projections
    for i in range(len(xvector)):
    # arrows project features (ie columns from csv) as vectors onto PC axes
        ax.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys), color = 'b', width = 0.0005, head_width = 0.0025)
        ax.text(xvector[i]*max(xs)*1, yvector[i]*max(ys)*1, list(data.columns.values)[i], color = 'r')

    for i in range(len(xs)):
    # circles project documents (ie rows from csv) as points onto PC axes
        ax.text(xs[i]*1, ys[i]*1, list(data.index)[i], color = colors[i])
        ax.scatter(xs[i], ys[i], c = colors[i], s = 50)
        
    ax.legend(genotypes)
    ax.grid()



# pca_vector analysis and visualization
def LDA_analysis(df, file_path):
    
    x = df.loc[:, features].values
    
    y = df.loc[:,['genotype']].values
    
    X = StandardScaler().fit_transform(x)
    
    print (pd.DataFrame(data = x, columns = features).head())
    
    data = pd.DataFrame(data = X, columns = features)
    
    n_components = len(data.columns)
    
    pca = PCA(n_components = n_components)
    
    pcafit = pca.fit_transform(X)
    
    # Percentage of variance explained for each components
    
    print('explained variance ratio (first two components): {0}'.format(str(pca.explained_variance_ratio_ )))
  
  
    
    lda = LinearDiscriminantAnalysis(n_components = n_components)
    
    X_r2 = lda.fit(X, y).transform(X)
    
    
    #print(principalComponents.shape)


def label_bars(ax, bars, text_format, **kwargs):
    """
    Attaches a label on every bar of a regular or horizontal bar chart
    """
    ys = [bar.get_y() for bar in bars]
    y_is_constant = all(y == ys[0] for y in ys)  # -> regular bar chart, since all all bars start on the same y level (0)

    if y_is_constant:
        _label_bar(ax, bars, text_format, **kwargs)
    else:
        _label_barh(ax, bars, text_format, **kwargs)


def _label_bar(ax, bars, text_format, **kwargs):
    """
    Attach a text label to each bar displaying its y value
    """
    max_y_value = ax.get_ylim()[1]
    inside_distance = max_y_value * 0.01
    outside_distance = max_y_value * 0.01

    for bar in bars:
        text = text_format.format(bar.get_height())
        text_x = bar.get_x() + bar.get_width() / 2

        is_inside = bar.get_height() >= max_y_value * 0.15
        if is_inside:
            color = "orange"
            text_y = bar.get_height() - inside_distance
        else:
            color = "orange"
            text_y = bar.get_height() + outside_distance

        ax.text(text_x, text_y, text, ha='center', va='bottom', color=color, **kwargs)


def _label_barh(ax, bars, text_format, **kwargs):
    """
    Attach a text label to each bar displaying its y value
    Note: label always outside. otherwise it's too hard to control as numbers can be very long
    """
    max_x_value = ax.get_xlim()[1]
    distance = max_x_value * 0.0025

    for bar in bars:
        text = text_format.format(bar.get_width())

        text_x = bar.get_width() + distance
        text_y = bar.get_y() + bar.get_height() / 2

        ax.text(text_x, text_y, text, va='center', **kwargs)


#colormap mapping
def get_cmap(n):
    """get the color mapping""" 
    #viridis, BrBG, hsv, copper
    #Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    #RGB color; the keyword argument name must be a standard mpl colormap name
    
    
    #colors = plt.cm.hsv(np.linspace(0, 1, n))
    
    #colors = plt.cm.viridis(np.linspace(0, 1, n))
    
    colors = plt.cm.tab20(np.linspace(0, 1, n))
    
    colors = np.atleast_2d(colors)
    
    return colors
    
    #return plt.cm.get_cmap(name,n+1)


# pca_vector analysis and visualization
def pca_analysis(df, file_path):
    
    x = df.loc[:, features].values
    
    y = df.loc[:,['genotype']].values
    
    X = StandardScaler().fit_transform(x)
    
    #print pd.DataFrame(data = x, columns = features).head()
    
    data = pd.DataFrame(data = X, columns = features)
    
    n_components = len(data.columns)
    
    #n_components = 5
    
    print("n_components = {}".format(n_components))
    
    pcamodel = PCA(n_components = n_components)
    
    
    principalComponents = pcamodel.fit_transform(X)
    
    #print(principalComponents.shape)
    
    #Cumulative Explained Variance
    ####################################################################
    fig, ax = plt.subplots(figsize = (16,9)) 
    #plt.figure(figsize = (16,9))
    
    bars = ax.bar(range(1,len(pcamodel.explained_variance_ratio_ )+1),pcamodel.explained_variance_ratio_ )
    
    ax.set_ylabel('Explained variance')
    ax.set_xlabel('Components')
    
    value_format = "{:.1%}"  # displaying values as percentage with one fractional digit
    
    label_bars(ax, bars, value_format)
                
    #plt.text(range(1,len(pcamodel.explained_variance_ratio_ )+1),np.cumsum(pcamodel.explained_variance_ratio_),  pcamodel.explained_variance_ratio_, color = 'red')
    
    ax.plot(range(1,len(pcamodel.explained_variance_ratio_ )+1),np.cumsum(pcamodel.explained_variance_ratio_),  c = 'red', label = "PCA Explained Variance")
    
    #plt.plot(range(1,len(pca.explained_variance_ )+1),np.cumsum(pca.explained_variance_),  c = 'red', label = "Cumulative Explained Variance")
    
    ax.legend(loc='upper left')
    
    result_file_path = file_path + '/'  + 'pca_cev.png'
    
    plt.savefig(result_file_path)
    
    print("pca.explained_variance_ratio_ = {0}".format(pcamodel.explained_variance_ratio_))

    plt.close()
    ####################################################################
    
    #Components heatmap
    ####################################################################
    fig = plt.figure(figsize = (20,20))
    
    #cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    cmap = 'viridis'
    
    # generating correlation heatmap 
    #ax = sns.heatmap(zoo_data.corr(), annot = True) 
 
    #font_size = 20
    
    #sns.set(font_scale = 1.0)
    
    ax = sns.heatmap(pcamodel.components_, cmap = cmap, 
                    yticklabels =[ "PCA"+str(x) for x in range(1,pcamodel.n_components_+1)],
                    xticklabels=list(data.columns), 
                    cbar_kws={"orientation": "horizontal"}, square = True, annot = True)
                    
    ax.set_aspect("equal")
    
    #ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = font_size)
                    
    #ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = font_size)
    
   
    result_file_path = file_path + '/'  + 'pca_heat.png'
    
    plt.savefig(result_file_path)

    plt.close()
    ####################################################################
    
    
    # define components by features
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 
                                                                      'principal component 2',
                                                                      'principal component 3',
                                                                      'principal component 4',
                                                                      'principal component 5',
                                                                      'principal component 6',
                                                                      'principal component 7',
                                                                      'principal component 8',
                                                                      'principal component 9',
                                                                      'principal component 10'])
                                                                      

    
     # 0,1 denote PC1 and PC2; change values for other PCs
    xvector = pcamodel.components_[0] 
    yvector = pcamodel.components_[1]

    xs = pcamodel.transform(data)[:,0] 
    ys = pcamodel.transform(data)[:,1]

    finalDf = pd.concat([principalDf, df[['genotype']]], axis = 1)
    #print finalDf.head()
    
    
    
    #main components vector map, PCA Biplot
    #############################################################
    fig = plt.figure(figsize = (68,68))
    
    #print ('Backend:',plt.get_backend())
    
    plt.rcParams.update({'font.size': 48})
    
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    
    ax = fig.add_subplot(1,1,1) 
    
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('PCA')
    
    
    # get unique genotype names 
    genotypes = list(set(df['genotype']))

    #colors = plt.cm.rainbow(np.linspace(0, 1, len(genotypes)))
    #colors = plt.cm.Spectral(np.linspace(0, 1, len(genotypes)))
    #colors = plt.cm.Spectral(np.linspace(0, 1, len(genotypes)))
    #colors = plt.cm.tab20(np.linspace(0, 1, len(genotypes)))

    colors = get_cmap(len(genotypes))
    

    #visualize points
    for idx, (genotype, color) in enumerate(zip(genotypes, colors)):
        
        #print(idx)
        
        indicesToKeep = finalDf['genotype'] == genotype
        
        ax.plot(finalDf.loc[indicesToKeep, 'principal component 1'].to_numpy(), finalDf.loc[indicesToKeep, 'principal component 2'].to_numpy(), c = color, marker = '.', linestyle = '', markersize = 85)
    
        #color_hsv = np.delete(color.copy, -1, axis=1)
        print("color value of {0} = {1}".format(genotype, rgb2hex(color)))
    

    
    #visualize vectors
    for i, color in zip(range(len(xvector)), colors):
        # arrows project features (ie columns from csv) as vectors onto PC axes
        
        #print(i)
        #ax.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys), color = color, width = 0.0005, head_width = 0.0025)
        
        ax.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys), color = 'b', width = 0.0005, head_width = 0.0025)
        
        #ax.arrow(0, 0, xvector[i]*np.mean(xs), yvector[i]*np.mean(ys), color = 'b', width = 0.0005, head_width = 0.0025)
        
        #texts = ax.text(xvector[i]*max(xs)*1.05, yvector[i]*max(ys)*1.05, list(data.columns.values)[i], color = colors[i])
        
        #adjust_text(texts)
    
    texts = [ax.text(xvector[i]*max(xs)*1.05, yvector[i]*max(ys)*1.05, '%s' %list(data.columns.values)[i], fontsize = 40) for i in range(len(xvector))]
    
    adjust_text(texts)
    
    #ax.legend(genotypes, labelspacing = 1.2, frameon = True, prop={'size': 30})
    
    lgd = ax.legend(genotypes, labelspacing = 1.2, frameon = True, prop={'size': 30}, loc='center left', bbox_to_anchor=(1, 0.5))
    
    
    ax.grid()
    
    result_file_path = file_path + '/'  + 'pca_map.png'
    
    plt.savefig(result_file_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    plt.close()
    




# ploty scatter plot, visualize the first two principal components of a PCA, by reducing a dataset of 4 dimensions to 2D.
def scatter_plot_2d(df):
    
    import plotly.express as px
    from sklearn.decomposition import PCA
    
    
    #df = px.data.iris()
    #X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    
    X = df[['root system diameter max',
                'root system diameter min',
                'root system diameter',
                'root system length',
                'root system angle',
                'root system angle max',
                'root system angle min',
                'root system volume',
                'root system eccentricity',
                'root system bushiness']]

    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    fig = px.scatter(components, x=0, y=1, color=df['genotype'])
    
    fig.show()
    
    result_file_path = file_path + '/'  + 'scatter_2d.html'
    
    fig.write_html(result_file_path)



def scatter_plot_3d(df):
    
    import plotly.express as px
    from sklearn.decomposition import PCA

    X = df[['root system diameter max',
                'root system diameter min',
                'root system diameter',
                'root system length',
                'root system angle',
                'root system angle max',
                'root system angle min',
                'root system volume',
                'root system eccentricity',
                'root system bushiness']]


    pca = PCA(n_components=3)
    components = pca.fit_transform(X)

    total_var = pca.explained_variance_ratio_.sum() * 100

    fig = px.scatter_3d(
        components, x=0, y=1, z=2, color=df['genotype'],
        title=f'Total Explained Variance: {total_var:.2f}%',
        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
    )
    fig.show()
    
    result_file_path = file_path + '/'  + 'scatter_3d.html'
    
    fig.write_html(result_file_path)






if __name__ == '__main__':
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True,    help = "path to image file")
    ap.add_argument("-f", "--filename", required = True, help = "data file name")
    args = vars(ap.parse_args())
    
    # setting path 
    file_path = args["path"]
    filename = args["filename"]
    data_file_path = file_path + filename
    
    #result_file_path = file_path + '/'  + 'pca_map.png'
    
    
    #data_file_path = file_path + 'trait_sum_pca.xlsx'

    
    
    # loading dataset into Pandas DataFrame

    df = pd.read_excel(data_file_path, names=['genotype',
                                            'root system diameter max',
                                            'root system diameter min',
                                            'root system diameter',
                                            'root system length',
                                            'root system angle',
                                            'root system angle max',
                                            'root system angle min',
                                            'root system volume',
                                            'root system eccentricity',
                                            'root system bushiness'])
    

    
    print (df.head())
    
    #aggregate traits
    features = ['root system diameter max',
                'root system diameter min',
                'root system diameter',
                'root system length',
                'root system angle',
                'root system angle max',
                'root system angle min',
                'root system volume',
                'root system eccentricity',
                'root system bushiness']

    
    #class_data = df['genotype']
    #class_names = dict(enumerate(df['genotype']))
    
    genotype_names = list(set(df['genotype']))
    
    
    print(type(features))
    print(type(genotype_names))
    
    print(genotype_names)
    
    '''
    import plotly.express as px
    from sklearn.decomposition import PCA

    df = px.data.iris()
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    fig = px.scatter(components, x=0, y=1, color=df['species'])
    fig.show()
    '''
    
    #################################################################
    
    pca_analysis(df, file_path)
    
    scatter_plot_2d(df)
    
    scatter_plot_3d(df)
    
    #LDA_analysis(df, file_path)
    
    #feature_distributation()
    
    
    ###############################################################
    '''
    fig = plt.figure(figsize = (68,68))
    
    #g = sns.pairplot(df, hue = 'genotype', markers = '*')
    
    g = sns.pairplot(df, hue = 'genotype')
    
    result_file_path = file_path + '/'  + 'pireplot.png'
    
    plt.savefig(result_file_path)
    
    
    # violinplot_
    for feature_value in features:
        
        #print feature_value
        #print(type(feature_value))
        
        fig = plt.figure(figsize = (9,9))
         
        g = sns.violinplot(x = 'genotype',  y = feature_value, data = df, inner = 'quartile')
        
        g.set_xticklabels(g.get_xticklabels(), rotation = 45)
        
        result_file_path = file_path + '/'  + 'violinplot_' + str(feature_value) + '.png'
        
        plt.savefig(result_file_path)
        
        plt.close()
    
    '''


    '''
    from sklearn.datasets import load_iris
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt


    #iris_obj = load_iris()
    
    iris_obj = df
    
    iris_df = pd.DataFrame(iris_obj.data, columns=iris_obj.features)

    iris_df["species"] = [iris_obj.Genotype[s] for s in iris_obj.Genotype]
    
    
    
    iris_df.head()

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=2)),])

    pca_data = pd.DataFrame(
        pipeline.fit_transform(iris_df.drop(columns=["species"])),
        columns=["PC1", "PC2"],
        index=iris_df.index,
    )
    pca_data["species"] = iris_df["species"]

    pca_step = pipeline.steps[1][1]
    loadings = pd.DataFrame(
        pca_step.components_.T,
        columns=["PC1", "PC2"],
        index=iris_df.drop(columns=["species"]).columns,
    )

    g = sns.scatterplot(data=pca_data, x="PC1", y="PC2", hue="species")

    # Add loadings
    loading_plot(loadings[["PC1", "PC2"]].values, loadings.index, scale=2, arrow_size=0.08)


    # Add variance explained by the
    g.set_xlabel(f"PC1 ({pca_step.explained_variance_ratio_[0]*100:.2f} %)")
    g.set_ylabel(f"PC2 ({pca_step.explained_variance_ratio_[1]*100:.2f} %)")

    
    result_file_path = file_path + '/'  + 'PCA_with_loadings.png'
    
    plt.savefig(result_file_path, bbox_inches='tight')
    '''
