import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os



def load_data(file1,file2=None):
    df1 = pd.read_csv(file1)
    if file2 is not None:
        df2 = pd.read_csv(file2)
    df1.columns=['iter','mean','std']
    df2.columns=df1.columns
    df1.insert(len(df1.columns),'condition','nodagger')
    df1.insert(len(df1.columns),'unit',0)
    df1.columns
    df2.insert(len(df2.columns),'condition','dagger')
    df2.insert(len(df2.columns),'unit',0)
    df = pd.concat([df1,df2],ignore_index=True)

    sns.tsplot(data=df,time='iter',value='mean',unit='unit',condition='condition')
    plt.legend(loc='best')
    #plt.show()

    figname = 'compare'
    plt.savefig(figname)


def main():
    filename1 ='Hopper-v120.csv'
    filename2 = 'Hopper-v1dagger20'
    load_data(filename1,filename2)

if __name__==  "__main__":
    main()
