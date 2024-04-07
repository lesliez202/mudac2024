import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings 
import scipy.stats
warnings.simplefilter("ignore", category=FutureWarning)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


df = pd.read_csv('C:\\Users\\zacha\\Desktop\\MUDAC_Code\\Final_Mudac.csv')

average_temp = [61.51428571, 61.52857143, 62.55714286, 63.27142857,
64.01428571, 63.22857143,61.07142857,62.02857143,
64.22857143,63.52857143, 64.62857143]
average_ppi_sb_april = [136.9, 149.900, 146.9, 144,9, 233.10, 146.8, 104.5, 87.4, 97.4, 117.8]
average_ppi_soy_april = [163.2, 224.5, 243.1, 242.5, 257.9, 164.5, 156.7, 154.8, 174.8, 145.8, 143.6]
average_ppi_corn_april = [137.6, 287.1, 258.7, 264.7, 199.2, 149.8, 141.900, 139.900, 148.4, 141.00, 126.0]

average_snowfall = [51.5, 77.1, 78.3, 32.0, 36.7, 32.4, 69.8, 67.7, 22.3, 86.6, 40.7]
average_oil = [79.48, 94.88, 94.05, 97.98, 93.17, 48.66, 43.29, 50.80, 65.23, 56.99, 39.16]
years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]

print(df.isna().sum())
def create_col(df, name, arr_vals, years):
    df[name] = 0.0
    for i in range(0,11):
        y = df['Year'] == years[i]
        df.loc[y, name] = arr_vals[i]   
    return df

create_col(df, 'Average Temp', average_temp, years)
create_col(df, 'PPI SB', average_ppi_sb_april, years)
create_col(df, 'PPI Soy', average_ppi_soy_april, years)
create_col(df, 'PPI Corn', average_ppi_corn_april, years)
create_col(df, 'Snowfall', average_snowfall, years)
create_col(df, 'Oil', average_oil, years)
df.to_csv('temp3.csv', index = False)

df['SUGARBEETS - ACRES PLANTED'].fillna(0.0, inplace = True)
df['SOYBEANS - ACRES PLANTED'].fillna(0.0, inplace = True)
df['CORN - ACRES PLANTED'].fillna(0.0, inplace = True)
print(df.isna().sum())
X = df[['Average Temp', 'PPI SB', 'PPI Soy', 'PPI Corn', 'Snowfall', 'Oil']]
print(X)
def MultiFunction(Dependent_variable, CI_Value, X):
    #Avoid numpy error
    X_col = X
    
    y = df[Dependent_variable]
    
    # Scale features using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Add constant term
    X = sm.add_constant(X)

    # Fit logistic regression model
    log_reg = sm.OLS(y, X).fit(method='qr', maxiter = 200)
    print(log_reg.summary())
    
    dataframe_1 = pd.DataFrame()
    #Creates dataframe column with independent variable names
    for i in range(0, len(X_col.columns)):
        new_frame = pd.DataFrame({'Independent Variable ' + Dependent_variable: [X_col.columns[i]]})
        dataframe_1 = pd.concat([dataframe_1, new_frame], axis = 0)   
    
    dataframe_2 = pd.DataFrame()
    #Coefficient column
    for j in range(0, len(X_col.columns)):
        new_frame = pd.DataFrame({'Cofficents ' + Dependent_variable: [round(np.exp(log_reg.params[j]),3)]})
        dataframe_2 = pd.concat([dataframe_2, new_frame], axis = 0)
    dataframe_1 = pd.concat([dataframe_1, dataframe_2], axis = 1)
    
    dataframe_3 = pd.DataFrame()
    #Lower CI column
    for k in range(0, len(X_col.columns)):
        new_frame = pd.DataFrame({'95 CI lower bound ' + Dependent_variable: [round(np.exp(log_reg.conf_int(alpha= CI_Value, cols=None)[0][k]),3)]})
        dataframe_3 = pd.concat([dataframe_3, new_frame], axis = 0)
    dataframe_1 = pd.concat([dataframe_1, dataframe_3], axis = 1)
    
    dataframe_4 = pd.DataFrame()
    for l in range(0, len(X_col.columns) ):
        new_frame = pd.DataFrame({'95 CI upper bound ' + Dependent_variable: [round(np.exp(log_reg.conf_int(alpha= CI_Value, cols=None)[1][l]),3)]})
        dataframe_4 = pd.concat([dataframe_4, new_frame], axis = 0)
    dataframe_1 = pd.concat([dataframe_1, dataframe_4], axis = 1)
    
    dataframe_1 = dataframe_1.sort_values(by=['Cofficents ' + Dependent_variable], ascending=False)
    return dataframe_1
sb = MultiFunction('CORN - ACRES PLANTED', 0.05, X)
print(sb)



def generate_table_pdf(df, description):
    fig, ax =plt.subplots(figsize=(12,4))
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=df.values,colLabels=df.columns,loc='center')
    pp = PdfPages(description + ".pdf")
    pp.savefig(fig, bbox_inches='tight')
    pp.close()
generate_table_pdf(sb, 'Reg Table')
