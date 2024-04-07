import pandas as pd
import numpy as np

df_AD = pd.read_csv("C:\\Users\\zacha\\Desktop\\Mudac_data\\AgDistricts.csv")
df_TL = pd.read_csv("C:\\Users\\zacha\\Desktop\\Mudac_data\\TillableLand.csv")
df_A = pd.read_csv("C:\\Users\\zacha\\Desktop\\Mudac_data\\Animals.csv")
df_FC = pd.read_csv("C:\\Users\\zacha\\Desktop\\Mudac_data\\FertilizerConsumption.csv")
df_Crops = pd.read_csv("C:\\Users\\zacha\\Desktop\\Mudac_data\\Crops.csv")
df_CPI = pd.read_csv("C:\\Users\\zacha\\Desktop\\Mudac_data\\CropProductivityIndex.csv")
arr = [df_AD, df_TL, df_A, df_FC, df_Crops, df_CPI]
for df in arr:
    print(df.shape)


def concatenating_function(df_1, df_2, join_1, join_2):
    arr = df_1.columns
    arr_1 = df_2.columns
    arr = np.concatenate([arr, arr_1], axis = 0)
    frame = pd.DataFrame()
    frames_list = []
    for i in range(0, len(df_1)):
        for j in range(0, len(df_2)):
            if (df_1.loc[i, join_1] == df_2.loc[j, join_2]):
                new_frame = pd.concat([df_1.loc[i], df_2.loc[j]], ignore_index = True)
                frames_list.append(new_frame)

    # Concatenate all data frames in the list
    frame = pd.concat([pd.DataFrame(frames_list), frame], axis = 0)
    frame.columns = arr
    print(frame)
    
    
    
    # Set a RangeIndex for the final data frame
    frame.index = pd.RangeIndex(start=0, stop=len(frame), step=1)
    return frame
'''
df_a = concatenating_function(df_A, df_CPI, 'ANSI of County_Animals', 'ANSI')
df_a = df_a[['State', 'County of County_Animals', 'ANSI',
       'Ag District', 'Ag District Code', 'Year of County_Animals',
       'CATTLE, COWS, BEEF - INVENTORY', 'CATTLE, COWS, MILK - INVENTORY',
       'CATTLE, INCL CALVES - INVENTORY', 'CATTLE, ON FEED - INVENTORY',
       'HOGS - INVENTORY', 'TURKEYS - INVENTORY','CPI', 'Notes']]
print(len(df_FC))
df_b = concatenating_function(df_a, df_FC, 'ANSI', 'ANSI')
df_b.to_csv('temp.csv', index = False)
print(df_b.columns)

print(df_b['ANSI'])
'''
df_b = pd.read_csv('C:\\Users\\zacha\\Desktop\\MUDAC_Code\\temp.csv')
df_c = concatenating_function(df_b, df_TL,'ANSI', 'ANSI')
df_c.to_csv('temp2.csv', index=False)
print(df_c.columns)
#df_b = df[[]]
#df_b = concatenating_function(df_b, df_TL, 'ANSI', 'ANSI')
#df_a = concatenating_function(df_A, df_Crops, 'ANSI', 'ANSI of County_Animals')
#print(df_.columns)
#df_b = df[[]]




#df = concatenating_function(df_A, df_CPI, 'ANSI of County_Animals', 'ANSI')
#df = concatenating_function(df_A, df_CPI, 'ANSI of County_Animals', 'ANSI')
