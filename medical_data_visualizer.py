import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df =  pd.read_csv("medical_examination.csv")
#print(df.dtypes)

# 2 overweight
bmi = df['weight']/pow(df['height']/100,2)
#print(bmi)

df.loc[ bmi > 25, 'overweight' ]  = 1
df.loc[ bmi <= 25, 'overweight' ] = 0


# 3 data handling
df.loc[ df['cholesterol'] <= 1 , 'cholesterol' ] = 0
df.loc[ df['cholesterol'] > 1 , 'cholesterol' ] = 1

df.loc[ df['gluc'] <= 1 , 'gluc' ] = 0
df.loc[ df['gluc'] > 1 , 'gluc' ] = 1

#print(df[df.loc[:,'gluc']== 1])
# 4
def draw_cat_plot():
    # 5 melt into long format
    df_cat = df.loc[:,['cardio','cholesterol','gluc','smoke','alco','active','overweight']].melt(id_vars=['cardio'])
    df_cat = df_cat.astype({'value': int})

    # 6 group and count for all feature
    df_cat['total'] = 1
    df_cat_long = df_cat.groupby(['cardio','variable','value'], as_index=False)['total'].count()
 
    #print(df_cat_long.columns)
    #print(df_cat_long)

    # 7
    import seaborn as sns
    sns.set_theme(font_scale=1.5)

    # 8
    fig = sns.catplot(
    data=df_cat_long, x="variable", y="total", hue='value', col="cardio",
    kind="bar", height=8
    ).figure


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df
    df_heat = df_heat[  (df['ap_lo'] <= df['ap_hi']) &
                        (df['height'] >= df['height'].quantile(0.025)) &
                        (df['height'] <= df['height'].quantile(0.975)) &
                        (df['weight'] >= df['weight'].quantile(0.025)) &
                        (df['weight'] <= df['weight'].quantile(0.975))
                     ]
    #print(df_heat.shape[0])
    #df_heat = df_heat[df['height'] >= df['height'].quantile(0.025) ]
    #print(df_heat.shape[0])
    #df_heat = df_heat[df['height'] <= df['height'].quantile(0.975) ]
    #print(df_heat.shape[0])
    #df_heat = df_heat[df['weight'] >= df['weight'].quantile(0.025) ]
    #print(df_heat.shape[0])
    #df_heat = df_heat[df['weight'] <= df['weight'].quantile(0.975) ]
    #print(df_heat.shape[0])

    # 12 produce a correlation matrix
    corr = df_heat.corr()
    #corr = round(corr,1)
    

    # 13 procduce a mask for heatmap
    mask = np.triu(np.ones(corr.shape))
    #print(mask)

    # 14 
    fig, ax = plt.subplots(figsize=(15, 10))

    ax.set_xticklabels(labels=[], fontsize=12)
    ax.set_yticklabels(labels=[], fontsize=12)
    ax.grid(visible=False)

    # 15
    #print(corr)
    #sns.set(font_scale = 0.2)
    hm = sns.heatmap(corr, mask=mask, cmap="coolwarm", 
                     square=True, linewidth=0.5, 
                     annot=True, fmt='.1f',  annot_kws={'size': 10},
                     vmin=-0.7, vmax=0.7)
    
    hm.set_facecolor('white')
    
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    
     
    # 16
    fig.savefig('heatmap.png')
    return fig
