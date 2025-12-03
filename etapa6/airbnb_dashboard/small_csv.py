import pandas as pd

archivo = "listings_clean_core_eda.csv" 
df = pd.read_csv(archivo)  
df_small = df.sample(5000, random_state=42)
df_small.to_csv("data/listings_dashboard_sample.csv", index=False)



print(df_small)